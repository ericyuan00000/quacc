"""Transition state recipes for universal machine-learned interatomic potentials."""

# from __future__ import annotations

# from importlib.util import find_spec
# from typing import TYPE_CHECKING

# import numpy as np
# from ase.mep import NEB
# from monty.dev import requires

# from quacc import change_settings, get_settings, job, strip_decorator
# from quacc.recipes.newtonnet.core import _add_stdev_and_hess, freq_job, relax_job
# from quacc.runners.ase import Runner, run_neb
# from quacc.schemas.ase import Summarize, summarize_neb_run
# from quacc.utils.dicts import recursive_dict_merge

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from ase.atoms import Atoms
from ase.mep.neb import NEB
import rmsd

from quacc import job
from quacc.recipes.mlp._base import pick_calculator
from quacc.runners.ase import Runner
from quacc.schemas.ase import Summarize
from quacc.utils.dicts import recursive_dict_merge

has_geodesic_interpolate = bool(find_spec("geodesic_interpolate"))

if has_geodesic_interpolate:
    from geodesic_interpolate.geodesic import Geodesic
    from geodesic_interpolate.interpolation import redistribute

from transbymep import optimize_MEP

if TYPE_CHECKING:
    from typing import Any, Literal

    from ase.atoms import Atoms

    from quacc.types import OptParams, OptSchema, RunSchema, NebSchema

@job
def interpolate_job(
    reactant_atoms: Atoms,
    product_atoms: Atoms,
    interpolate_params: dict[str, Any] | None = None,
) -> OptSchema:
    """
    Interpolate between two structures.

    Parameters
    ----------
    reactant_atoms
        Reactant Atoms object
    product_atoms
        Product Atoms object
    interpolation_method
        Method to use for interpolation
    interpolate_kwargs
        Additional kwargs for the interpolation method

    Returns
    -------
    OptSchema
        Dictionary of results from [quacc.schemas.ase.Summarize.run][].
        See the type-hint for the data structure.
    """
    interpolate_defaults = {"alignment_method": "kabsch", "interpolation_method": "geodesic", "n_images": 10}
    interpolate_flag = recursive_dict_merge(interpolate_defaults, interpolate_params)

    alignment_method = interpolate_flag.pop("alignment_method")
    interpolation_method = interpolate_flag.pop("interpolation_method")
    n_images = interpolate_flag.pop("n_images")

    # if alignment_method == "kabsch":
    #     product_atoms.set_positions(rmsd.kabsch_rotate(product_atoms.get_positions(), reactant_atoms.get_positions()))

    if interpolation_method == "geodesic":
        images = geodesic_interpolate_wrapper(
            reactant_atoms, product_atoms, n_images, **interpolate_flag
        )
    else:
        images = [reactant_atoms]
        images += [
            reactant_atoms.copy() for i in range(n_images - 2)
        ]
        images += [product_atoms]
        neb = NEB(images)
        # Interpolate linearly the positions of the middle images:
        neb.interpolate(method=interpolation_method, **interpolate_flag)
        images = neb.images

    return {
        "initial_images": [reactant_atoms, product_atoms],
        "interpolated_images": images,
        "alignment_method": alignment_method,
        "interpolation_method": interpolation_method,
        "n_images": n_images,
    } | interpolate_flag

@job
def neb_job(
    images: list[Atoms],
    method: Literal["mace-mp-0", "mace-off", "m3gnet", "chgnet", "newtonnet"],
    neb_params: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    run_params: dict[str, Any] | None = None,
    **calc_kwargs,
) -> NebSchema:
    neb_defaults = {"climb": True}
    neb_flags = recursive_dict_merge(neb_defaults, neb_params)

    opt_defaults = {}
    opt_flags = recursive_dict_merge(opt_defaults, opt_params)

    run_defaults = {"fmax": 0.05}
    run_flags = recursive_dict_merge(run_defaults, run_params)
    
    calc = pick_calculator(method, **calc_kwargs)
    additional_fields = {"neb_flags": neb_flags, "opt_flags": opt_flags, "run_flags": run_flags}

    dyn = Runner(images, calc).run_neb(neb_flags, opt_flags, run_flags)

    return Summarize(
        additional_fields={"name": f"{method} NEB"} | additional_fields
    ).neb(dyn)

    # return {
    #     "initial_images": images,
    #     "neb_results": summarize_neb_run(
    #         dyn,
    #         n_images=len(images),
    #         additional_fields={
    #             "name": f"{method} NEB",
    #             "method": method,
    #             "neb_flags": neb_flags,
    #             "opt_flags": opt_flags,
    #             "run_flags": run_flags,
    #         },
    #     ),
    # }

@job
def pathopt_job(
    images: list[Atoms],
    potential_params: dict[str, Any] | None = None,
    path_params: dict[str, Any] | None = None,
    integrator_params: dict[str, Any] | None = None,
    optimizer_params: dict[str, Any] | None = None,
    num_optimizer_iterations: int = 1000,
):
    pathopt_params = {
        "images": images,
        "potential_params": potential_params,
        "path_params": path_params,
        "integrator_params": integrator_params,
        "optimizer_params": optimizer_params,
        "num_optimizer_iterations": num_optimizer_iterations,
    }
    output = pathopt_wrapper(**pathopt_params)
    
    output["initial_images"] = images
    initial_path = [images[0].copy() for _ in range(len(output["geometry"][0]))]
    for geom, atoms in zip(output["geometry"][0], initial_path):
        atoms.set_positions(geom.reshape(-1, 3))
    output["initial_path"] = initial_path
    output["initial_path_results"] = {
        "energy": output["energy"][0],
        "force": output["force"][0],
        "velocity": output["velocity"][0],
        "integral": output["integral"][0],
    }
    final_path = [images[0].copy() for _ in range(len(output["geometry"][-1]))]
    for geom, atoms in zip(output["geometry"][-1], final_path):
        atoms.set_positions(geom.reshape(-1, 3))
    output["final_path"] = final_path
    output["final_path_results"] = {
        "energy": output["energy"][-1],
        "force": output["force"][-1],
        "velocity": output["velocity"][-1],
        "integral": output["integral"][-1],
    }

    output.pop("geometry")
    output.pop("energy")
    output.pop("velocity")
    output.pop("force")

    return output | pathopt_params


def geodesic_interpolate_wrapper(
    reactant: Atoms,
    product: Atoms,
    n_images: int = 10,
    perform_sweep: bool | Literal["auto"] = "auto",
    redistribute_tol: float = 1e-2,
    smoother_tol: float = 2e-3,
    max_iterations: int = 15,
    max_micro_iterations: int = 20,
    morse_scaling: float = 1.7,
    geometry_friction: float = 1e-2,
    distance_cutoff: float = 3.0,
    sweep_cutoff_size: int = 35,
) -> list[Atoms]:
    """
    Interpolates between two geometries and optimizes the path with the geodesic method.

    Parameters
    ----------
    reactant
        The ASE Atoms object representing the initial geometry.
    product
        The ASE Atoms object representing the final geometry.
    n_images
        Number of images for interpolation. Default is 10.
    perform_sweep
        Whether to sweep across the path optimizing one image at a time.
        Default is to perform sweeping updates if there are more than 35 atoms.
    redistribute_tol
        the value passed to the tol keyword argument of
         geodesic_interpolate.interpolation.redistribute. Default is 1e-2.
    smoother_tol
        the value passed to the tol keyword argument of geodesic_smoother.smooth
        or geodesic_smoother.sweep. Default is 2e-3.
    max_iterations
        Maximum number of minimization iterations. Default is 15.
    max_micro_iterations
        Maximum number of micro iterations for the sweeping algorithm. Default is 20.
    morse_scaling
        Exponential parameter for the Morse potential. Default is 1.7.
    geometry_friction
        Size of friction term used to prevent very large changes in geometry. Default is 1e-2.
    distance_cutoff
        Cut-off value for the distance between a pair of atoms to be included in the coordinate system. Default is 3.0.
    sweep_cutoff_size
        Cut off system size that above which sweep function will be called instead of smooth
        in Geodesic.

    Returns
    -------
    list[Atoms]
        A list of ASE Atoms objects representing the smoothed path between the reactant and product geometries.
    """
    reactant = reactant.copy()
    product = product.copy()

    # Read the initial geometries.
    chemical_symbols = reactant.get_chemical_symbols()

    # First redistribute number of images. Perform interpolation if too few and subsampling if too many images are given
    raw_interpolated_positions = redistribute(
        chemical_symbols,
        [reactant.positions, product.positions],
        n_images,
        tol=redistribute_tol,
    )

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    geodesic_smoother = Geodesic(
        chemical_symbols,
        raw_interpolated_positions,
        morse_scaling,
        threshold=distance_cutoff,
        friction=geometry_friction,
    )
    if perform_sweep == "auto":
        perform_sweep = len(chemical_symbols) > sweep_cutoff_size
    if perform_sweep:
        geodesic_smoother.sweep(
            tol=smoother_tol, max_iter=max_iterations, micro_iter=max_micro_iterations
        )
    else:
        geodesic_smoother.smooth(tol=smoother_tol, max_iter=max_iterations)
    return [
        Atoms(symbols=chemical_symbols, positions=geom)
        for geom in geodesic_smoother.path
    ]

def pathopt_wrapper(**pathopt_params):
    """
    Optimize a path of images using a machine-learned potential.

    Parameters
    ----------
    images
        List of Atoms objects representing the path of images.
    method
        The method to use for the optimization.
    opt_params
        Additional parameters for the optimization method.
    calc_kwargs
        Additional parameters for the calculator.

    Returns
    -------
    dict
        Dictionary containing the initial images, the optimized images, and the optimization results.
    """
    paths_geometry, paths_energy, paths_velocity, paths_force, paths_integral, paths_neval = optimize_MEP(**pathopt_params)
    return {
        "geometry": paths_geometry,
        "energy": paths_energy,
        "velocity": paths_velocity,
        "force": paths_force,
        "integral": paths_integral,
        "neval": paths_neval,
    }