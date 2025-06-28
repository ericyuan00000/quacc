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
    from ase.calculators.calculator import Calculator

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
    # interpolate_defaults = {"interpolation_method": "geodesic", "n_images": 10}
    interpolate_defaults = {"interpolation_method": "popcornn", "n_images": 10}
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
    elif interpolation_method == "popcornn":
        import torch
        from popcornn import Popcornn
        
        mep = Popcornn(
            images=[reactant_atoms, product_atoms], 
            num_record_points=n_images, 
            **interpolate_params.get('init_params'),
        )
        images, _ = mep.optimize_path(*interpolate_params.get('opt_params'))
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
def find_highest_job(
    images: list[Atoms],
    method: Literal["mace-mp-0", "mace-off", "m3gnet", "chgnet", "newtonnet", "uma"],
    **calc_kwargs,
) -> dict[str, Any]:
    """
    Find the highest energy image in a list of images.

    Parameters
    ----------
    images
        List of Atoms objects representing the images.
    calc_kwargs
        Additional keyword arguments for the calculator.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the highest energy image and its energy.
    """
    calc = pick_calculator(method, **calc_kwargs)
    
    for image in images:
        image.calc = calc
        image.get_potential_energy()
        image.calc = SinglePointCalculator(image, **image.calc.results)

    energies = [image.get_potential_energy() for image in images]
    highest_index = np.argmax(energies)
    
    return {
        "atoms": images[highest_index],
        "results": images[highest_index].calc.results,
        "trajectory": images,
        "trajectory_results": [image.calc.results for image in images],
    }

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
def popcornn_job(
    images: list[Atoms],
    init_params: dict[str] = {},
    opt_params: list[dict[str, Any]] = [],
):
    import torch
    from popcornn import Popcornn
    
    mep = Popcornn(images=images, **init_params)
    final_trajectory, ts_atoms = mep.optimize_path(*opt_params)
    final_trajectory_results = [atoms.calc.results for atoms in final_trajectory]
    ts_atoms_results = ts_atoms.calc.results
    output = {
        "initial_trajectory": images,
        "final_trajectory": final_trajectory,
        "final_trajectory_results": final_trajectory_results,
        "atoms": ts_atoms,
        "results": ts_atoms_results,
        "parameters": {'init_params': init_params, 'opt_params': opt_params},
    }

    return output


@job
def gsm_job(
    reactant_atoms: Atoms,
    product_atoms: Atoms,
    method: Literal["mace-mp-0", "mace-off", "m3gnet", "chgnet", "newtonnet"],
    gsm_params: dict[str, Any] | None = None,
    **calc_kwargs,
) -> OptSchema:
    """
    Run the GSM algorithm to find the transition state.

    Parameters
    ----------
    reactant
        Reactant Atoms object
    product
        Product Atoms object
    method
        Universal ML interatomic potential method to use
    optimizer_method
        Optimizer method to use for the GSM algorithm
    coordinate_type
        Coordinate type to use for the GSM algorithm
    line_search
        Line search method to use for the GSM algorithm
    only_climb
        Whether to only climb the GSM path or not
    step_size_cap
        Step size cap for the GSM algorithm
    num_nodes
        Number of nodes for the GSM algorithm
    add_node_tol
        Tolerance for adding new nodes in the GSM algorithm
    conv_tol
        Convergence tolerance for the GSM algorithm
    conv_Ediff
        Energy difference convergence for the GSM algorithm
    conv_gmax
        Max grad rms threshold for the GSM algorithm
    max_gsm_iterations
        Maximum number of iterations for the GSM algorithm

    Returns
    -------
    OptSchema
        Dictionary of results from [quacc.schemas.ase.Summarize.run][].
        See the type-hint for the data structure.
    """
    
    calc = pick_calculator(method, **calc_kwargs)
    
    images, ts_atoms = de_gsm_wrapper(
        atoms_reactant=reactant_atoms, 
        atoms_product=product_atoms, 
        calculator=calc, 
        **gsm_params,
    )

    return {
        "initial_images": [reactant_atoms, product_atoms],
        "atoms": ts_atoms,
        "results": ts_atoms.calc.results,
        "final_trajectory": images,
        "final_trajectory_results": [atoms.calc.results for atoms in images],
    } | gsm_params


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
        custom_hessian = opt_flags["optimizer_kwargs"].pop("custom_hessian", False)
        if custom_hessian:
            def get_hessian(atoms):
                if "hessian" in atoms.calc.results:
                    hessian = atoms.calc.results["hessian"]
                else:
                    hessian = atoms.calc.get_hessian(atoms)
                hessian = hessian.reshape(len(atoms) * 3, len(atoms) * 3)
                return hessian
            opt_flags["optimizer_kwargs"]["hessian_function"] = get_hessian
            # calc_kwargs["properties"] = ('energy', 'forces', 'hessian')
            calc_kwargs["calculate_hessian"] = True

    calc = pick_calculator(method, **calc_kwargs)

    dyn = Runner(image, calc).run_opt(**opt_flags)

    return Summarize(
        additional_fields={"name": f"{method} IRC"} | (additional_fields or {})
    ).opt(dyn, check_convergence=False)


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

def de_gsm_wrapper(
        atoms_reactant: Atoms,
        atoms_product: Atoms,
        calculator: Calculator,
        optimizer_method = "eigenvector_follow",
        coordinate_type = "TRIC",
        line_search = 'NoLineSearch',  # OR: 'backtrack'
        only_climb = False,
        step_size_cap = 0.1,  # DMAX in the other wrapper
        num_nodes = 9,  # 20 for SE-GSM
        add_node_tol = 0.1,  # convergence for adding new nodes
        conv_tol = 0.001,  # Convergence tolerance for optimizing nodes
        conv_Ediff = 100.,  # Energy difference convergence of optimization.
        conv_gmax = 100.,  # Max grad rms threshold
        max_gsm_iterations = 100,
        max_opt_steps = 5,  # 20 for SE-GSM
):
    import os
    from copy import deepcopy
    from pyGSM.level_of_theories.ase import ASELoT
    from pyGSM.potential_energy_surfaces import PES
    from pyGSM.utilities.elements import ElementData
    from pyGSM.coordinate_systems.topology import Topology
    from pyGSM.coordinate_systems.primitive_internals import PrimitiveInternalCoordinates
    from pyGSM.coordinate_systems.delocalized_coordinates import DelocalizedInternalCoordinates
    from pyGSM.molecule import Molecule
    from pyGSM.optimizers.eigenvector_follow import eigenvector_follow
    from pyGSM.optimizers.lbfgs import lbfgs
    from pyGSM.growing_string_methods import DE_GSM

    # Level of theory
    lot = ASELoT.from_options(
        calculator=calculator, 
        geom=[[atom.symbol, *atom.position] for atom in atoms_reactant],
    )

    # Potential energy surface
    pes_obj = PES.from_options(
        lot=lot,
    )

    # Build the topology
    element_table = ElementData()
    elements = [element_table.from_symbol(sym) for sym in atoms_reactant.get_chemical_symbols()]
    topology_reactant = Topology.build_topology(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
    )
    topology_product = Topology.build_topology(
        xyz=atoms_product.get_positions(),
        atoms=elements,
    )

    # Union of bonds  # TODO: check if needed or not
    for bond in topology_product.edges():
        if bond in topology_reactant.edges() or bond[::-1] in topology_reactant.edges():
            continue
        if bond[0] > bond[1]:
            topology_reactant.add_edge(bond[0], bond[1])
        else:
            topology_reactant.add_edge(bond[1], bond[0])

    # Primitive internal coordinates
    prim_reactant = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        topology=topology_reactant,
        connect=(coordinate_type == "DLC"),
        addtr=(coordinate_type == "TRIC"),
        addcart=(coordinate_type == "HDLC"),
    )
    prim_product = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_product.get_positions(),
        atoms=elements,
        topology=topology_reactant,
        connect=(coordinate_type == "DLC"),
        addtr=(coordinate_type == "TRIC"),
        addcart=(coordinate_type == "HDLC"),
    )

    # Add product coordinates to reactant coordinates
    prim_reactant.add_union_primitives(prim_product)

    # Delocalized internal coordinates
    deloc_coords_reactant = DelocalizedInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        connect=(coordinate_type == "DLC"),
        addtr=(coordinate_type == "TRIC"),
        addcart=(coordinate_type == "HDLC"),
        primitives=prim_reactant,
    )

    # Set up the molecule
    molecule_reactant = Molecule.from_options(
        geom=[[atom.symbol, *atom.position] for atom in atoms_reactant],
        PES=pes_obj,
        coord_obj=deloc_coords_reactant,
        Form_Hessian=(optimizer_method == "eigenvector_follow"),
    )
    molecule_product = Molecule.copy_from_options(
        molecule_reactant,
        xyz=atoms_product.get_positions(),
        new_node_id=num_nodes - 1,
        copy_wavefunction=False,
    )

    # Set up the optimizer
    if optimizer_method == "eigenvector_follow":
        optimizer_object = eigenvector_follow.from_options(
            print_level=1,
            Linesearch=line_search,
            update_hess_in_bg=(not only_climb),
            conv_Ediff=conv_Ediff,
            conv_gmax=conv_gmax,
            DMAX=step_size_cap,
            opt_climb=only_climb,
        )
    elif optimizer_method == "lbfgs":
        optimizer_object = lbfgs.from_options(
            print_level=1,
            Linesearch=line_search,
            update_hess_in_bg=False,
            conv_Ediff=conv_Ediff,
            conv_gmax=conv_gmax,
            DMAX=step_size_cap,
            opt_climb=only_climb,
        )

    # Set up the growing string method
    gsm = DE_GSM.from_options(
        reactant=molecule_reactant,
        product=molecule_product,
        nnodes=num_nodes,
        CONV_TOL=conv_tol,
        CONV_gmax=conv_gmax,
        CONV_Ediff=conv_Ediff,
        ADD_NODE_TOL=add_node_tol,
        growth_direction=0,  # normal/react/prod: 0/1/2
        optimizer=optimizer_object,
        print_level=1,
        interp_method="DLC",
    )

    # Do growing string method
    gsm.go_gsm(max_gsm_iterations, max_opt_steps, rtype=(1 if only_climb else 2))

    # Get the path
    frames = []
    for geom in gsm.geometries:
        atoms = Atoms(symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom])
        atoms.calc = calculator
        atoms.calc = SinglePointCalculator(
            atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
        )
        frames.append(atoms)
    ts_atoms = frames[gsm.TSnode]

    # Clean up
    os.system(f"rm scratch/growth_iters_{gsm.ID:03d}_*.xyz")
    os.system(f"rm scratch/opt_iters_{gsm.ID:03d}_*.xyz")

    return frames, ts_atoms