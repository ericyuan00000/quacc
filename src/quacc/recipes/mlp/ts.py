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

import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.mep.neb import NEB, NEBOptimizer, interpolate
import rmsd

from quacc import job
from quacc.recipes.mlp._base import pick_calculator
from quacc.runners.ase import Runner
from quacc.schemas.ase import Summarize
from quacc.schemas.atoms import metadata_to_atoms
from quacc.utils.dicts import recursive_dict_merge

has_geodesic_interpolate = bool(find_spec("geodesic_interpolate"))

has_transbymep = bool(find_spec("transbymep"))


if TYPE_CHECKING:
    from typing import Any, Literal

    from ase.atoms import Atoms

    from quacc.types import OptParams, OptSchema, RunSchema, NebSchema

# @job
# def 


def interpolate_images(images_orig, n):
    images = images_orig.copy()
    for i in np.arange(len(images)-1)[::-1]:
        sub_images = [images[i].copy()] + [images[i+1].copy() for _ in range(n)]
        interpolate(sub_images)
        images = images[:i+1] + sub_images[1:-1] + images[i+1:]
    for i in range(len(images_orig)):
        assert np.allclose(images_orig[i].get_positions(), images[i * n].get_positions())
    return images

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
    # interpolate_defaults = {"alignment_method": "kabsch", "interpolation_method": "geodesic", "n_images": 10}
    interpolate_defaults = {"interpolation_method": "geodesic", "n_images": 10}
    interpolate_flag = recursive_dict_merge(interpolate_defaults, interpolate_params)

    # alignment_method = interpolate_flag.pop("alignment_method")
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
        # "alignment_method": alignment_method,
        "interpolation_method": interpolation_method,
        "n_images": n_images,
    } | interpolate_flag

@job
def neb_job(
    images: list[Atoms],
    method: Literal["mace-mp-0", "mace-off", "m3gnet", "chgnet", "newtonnet"],
    interpolate_n: int = None,
    neb_params: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    run_params: dict[str, Any] | None = None,
    **calc_kwargs,
) -> NebSchema:
    neb_defaults = {"climb": True, "neb": NEB}
    neb_flags = recursive_dict_merge(neb_defaults, neb_params)
    neb = neb_flags.pop("neb")
    if interpolate_n is not None:
        images = interpolate_images(images, interpolate_n)
    images = neb(images, **neb_flags)

    opt_defaults = {"optimizer": NEBOptimizer}
    opt_flags = recursive_dict_merge(opt_defaults, opt_params)
    opt = opt_flags.pop("optimizer")
    if isinstance(opt, str):
        if opt == "NEBOptimizer":
            opt = NEBOptimizer
        elif opt.startswith("Precon"):
            import ase.optimize.precon
            opt = getattr(ase.optimize.precon, opt)
        elif opt.startswith("SciPy"):
            import ase.optimize.sciopt
            opt = getattr(ase.optimize.sciopt, opt)
        else:
            import ase.optimize
            opt = getattr(ase.optimize, opt)

    run_defaults = {"fmax": 0.05}
    run_flags = recursive_dict_merge(run_defaults, run_params)
    
    calc = pick_calculator(method, **calc_kwargs)
    additional_fields = {"neb_flags": neb_flags, "opt_flags": opt_flags, "run_flags": run_flags}

    # dyn = Runner(images, calc).run_neb(neb, opt, neb_flags, opt_flags, run_flags)
    dyn = Runner(images, calc).run_neb(opt, opt_flags, run_flags)

    return Summarize(
        additional_fields={"name": f"{method} NEB"} | additional_fields
    ).neb(dyn, check_convergence=False)
    # ).neb(dyn, n_images=len(images))

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
    path_params: dict[str, Any] | None = None,
    integrator_params: dict[str, Any] | None = None,
    optimizer_params: dict[str, Any] | None = None,
    device: str = "cpu",
    num_optimizer_iterations: int = 1000,
    n_recording_frames: int | None = None,
    **potential_params,
):
    import torch
    from transbymep import optimize_MEP

    def output_to_atoms(path, time):
        images = []
        path_output = path(time, return_velocity=True, return_energy=True, return_force=True)
        for i in range(len(time)):
            atoms = Atoms(
                numbers=path.numbers.cpu().numpy(),
                positions=path_output.path_geometry[i].detach().cpu().numpy().reshape(-1, 3),
                velocities=path_output.path_velocity[i].detach().cpu().numpy().reshape(-1, 3),
                cell=path.cell.cpu().numpy(),
                pbc=path.pbc.cpu().numpy(),
            )
            calc = SinglePointCalculator(
                atoms=atoms,
                energy=path_output.path_energy[i].detach().item(),
                forces=path_output.path_force[i].detach().cpu().numpy().reshape(-1, 3),
            )
            atoms.calc = calc
            images.append(atoms)
        return images
    
    potential_params["potential"] = potential_params.pop("method")
    pathopt_params = {
        "images": images,
        "potential_params": potential_params,
        "path_params": path_params,
        "integrator_params": integrator_params,
        "optimizer_params": optimizer_params,
        "device": device,
        "num_optimizer_iterations": num_optimizer_iterations,
    }
    path = optimize_MEP(**pathopt_params)

    final_trajectory = output_to_atoms(
        path, 
        torch.linspace(
            path.t_init.item(), 
            path.t_final.item(), 
            n_recording_frames or len(images), 
            device=device,
        ),
    )
    final_trajectory_results = [atoms.calc.results for atoms in final_trajectory]
    ts_atoms = output_to_atoms(
        path, 
        path.TS_time,
    )[0]
    ts_atoms_results = ts_atoms.calc.results
    output = {
        "initial_trajectory": images,
        "final_trajectory": final_trajectory,
        "final_trajectory_results": final_trajectory_results,
        "atoms": ts_atoms,
        "results": ts_atoms_results,
        "parameters": pathopt_params,
    }
    
    return output


@job
def irc_job(
    image: Atoms,
    method: Literal["mace-mp-0", "mace-off", "m3gnet", "chgnet", "newtonnet"],
    direction: str,
    opt_params: dict[str, Any] | None = None,
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> OptSchema:
    opt_defaults = {"optimizer": "SellaIRC", "run_kwargs": {"direction": direction}}
    opt_flags = recursive_dict_merge(opt_defaults, opt_params)
    if opt_flags["optimizer"] == "SellaIRC":
        from sella import IRC
        opt_flags["optimizer"] = IRC

    calc = pick_calculator(method, **calc_kwargs)

    dyn = Runner(image, calc).run_opt(**opt_flags)

    return Summarize(
        additional_fields={"name": f"{method} IRC"} | (additional_fields or {})
    ).opt(dyn)


def geodesic_interpolate_wrapper(
    reactant: Atoms,
    product: Atoms,
    n_images: int = 17,
    sweep: bool = None,
    tol: float = 2e-3,
    maxiter: int = 15,
    microiter: int = 20,
    scaling: float = 1.7,
    friction: float = 1e-2,
    dist_cutoff: float = 3,
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
    import numpy as np
    from geodesic_interpolate.geodesic import Geodesic
    from geodesic_interpolate.interpolation import redistribute
    np.random.seed(0)

    reactant = reactant.copy()
    product = product.copy()

    # Read the initial geometries.
    symbols = reactant.get_chemical_symbols()

    # First redistribute number of images. Perform interpolation if too few and subsampling if too many images are given
    raw = redistribute(symbols, [reactant.positions, product.positions], n_images, tol=tol * 5)

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    smoother = Geodesic(symbols, raw, scaling, threshold=dist_cutoff, friction=friction)
    if sweep is None:
        sweep = len(symbols) > 35
    if sweep:
        smoother.sweep(tol=tol, max_iter=maxiter, micro_iter=microiter)
    else:
        smoother.smooth(tol=tol, max_iter=maxiter)
    return [Atoms(symbols=symbols, positions=geom) for geom in smoother.path]
