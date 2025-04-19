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
        output = popcornn_wrapper(
            images=[reactant_atoms, product_atoms], 
            num_record_points=n_images, 
            **interpolate_flag,
        )
        images = output["final_trajectory"]
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
def popcornn_job(
    images: list[Atoms],
    path_params: dict[str] = {},
    integrator_params: dict[str] = {},
    optimizer_params: dict[str] = {},
    num_optimizer_iterations: int = 1001,
    num_record_points: int | None = 101,
    device: str = "cpu",
    **potential_params,
):
    
    potential_params["potential"] = potential_params.pop("method")
    pathopt_params = {
        "images": images,
        "potential_params": potential_params,
        "path_params": path_params,
        "integrator_params": integrator_params,
        "optimizer_params": optimizer_params,
        "num_optimizer_iterations": num_optimizer_iterations,
        "num_record_points": num_record_points,
        "device": device,
    }
    output = popcornn_wrapper(**pathopt_params)
    
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
    ).opt(dyn, check_convergence=False)


def popcornn_wrapper(**pathopt_params):

    import torch
    from popcornn import optimize_MEP
    torch.cuda.empty_cache()
    
    final_trajectory, ts_atoms = optimize_MEP(**pathopt_params)
    final_trajectory_results = [atoms.calc.results for atoms in final_trajectory]
    ts_atoms_results = ts_atoms.calc.results
    output = {
        "initial_trajectory": pathopt_params.get('images'),
        "final_trajectory": final_trajectory,
        "final_trajectory_results": final_trajectory_results,
        "atoms": ts_atoms,
        "results": ts_atoms_results,
        "parameters": pathopt_params,
    }

    return output


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

# def de_gsm_wrapper(
#         atoms_reactant: Atoms,
#         atoms_product: Atoms,
#         calculator: Calculator,
#         optimizer_method = "eigenvector_follow",
#         coordinate_type = "TRIC",
#         line_search = 'NoLineSearch',  # OR: 'backtrack'
#         only_climb = False,
#         step_size_cap = 0.1,  # DMAX in the other wrapper
#         num_nodes = 9,  # 20 for SE-GSM
#         add_node_tol = 0.1,  # convergence for adding new nodes
#         conv_tol = 0.001,  # Convergence tolerance for optimizing nodes
#         conv_Ediff = 100.,  # Energy difference convergence of optimization.
#         conv_gmax = 100.,  # Max grad rms threshold
#         ID = 0,
#         nproc=1,
#         max_gsm_iterations = 100,
#         max_opt_steps = 5,  # 20 for SE-GSM
#         reparametrize=True,
#         start_climb_immediately=False,
#         fixed_reactant=False,
#         fixed_product=False,
#         restart_file=False
# ):
#     print('Parsed GSM')

#     # set up calculator
    
        
#     # LOT
#     if calc.lower() in ['chg','mattersim']:
#         cell = [100,100,100]
#     else:
#         cell = None
#     lot = ASELoT.from_options(calculator,
#                               nproc=nproc,
#                               geom=[[x.symbol, *x.position] for x in atoms_reactant],
#                               cell=cell,
#                               ID=ID)

#     # PES
#     pes_obj = PES.from_options(lot=lot, ad_idx=0, multiplicity=1)

#     # load the initial string
#     if restart_file:
#         geoms = manage_xyz.read_molden_geoms(restart_file)
#     else:
#         geoms = [atoms_reactant,atoms_product]
    
#     # Build the topology
#     nifty.printcool("Building the topologies")
#     element_table = ElementData()
#     elements = [element_table.from_symbol(sym) for sym in atoms_reactant.get_chemical_symbols()]

#     topology_reactant = Topology.build_topology(
#         xyz=atoms_reactant.get_positions(),
#         atoms=elements
#     )

#     topology_product = Topology.build_topology(
#         xyz=atoms_product.get_positions(),
#         atoms=elements
#     )

#     # Union of bonds
#     # debated if needed here or not
#     for bond in topology_product.edges():
#         if bond in topology_reactant.edges() or (bond[1], bond[0]) in topology_reactant.edges():
#             continue
#         print(" Adding bond {} to reactant topology".format(bond))
#         if bond[0] > bond[1]:
#             topology_reactant.add_edge(bond[0], bond[1])
#         else:
#             topology_reactant.add_edge(bond[1], bond[0])

#     # primitive internal coordinates
#     nifty.printcool("Building Primitive Internal Coordinates")
#     connect = False
#     addtr = False
#     addcart = False
#     if coordinate_type == "DLC":
#         connect = True
#     elif coordinate_type == "TRIC":
#         addtr = True
#     elif coordinate_type == "HDLC":
#         addcart = True

#     prim_reactant = PrimitiveInternalCoordinates.from_options(
#         xyz=atoms_reactant.get_positions(),
#         atoms=elements,
#         topology=topology_reactant,
#         connect=connect,
#         addtr=addtr,
#         addcart=addcart,
#     )

#     prim_product = PrimitiveInternalCoordinates.from_options(
#         xyz=atoms_product.get_positions(),
#         atoms=elements,
#         topology=topology_reactant,
#         connect=connect,
#         addtr=addtr,
#         addcart=addcart,
#     )

#     # add product coords to reactant coords
#     prim_reactant.add_union_primitives(prim_product)

#     # Delocalised internal coordinates
#     nifty.printcool("Building Delocalized Internal Coordinates")
#     deloc_coords_reactant = DelocalizedInternalCoordinates.from_options(
#         xyz=atoms_reactant.get_positions(),
#         atoms=elements,
#         connect=coordinate_type == "DLC",
#         addtr=coordinate_type == "TRIC",
#         addcart=coordinate_type == "HDLC",
#         primitives=prim_reactant
#     )

#     # Molecules
#     nifty.printcool("Building the reactant object with {}".format(coordinate_type))
#     from_hessian = optimizer_method == "eigenvector_follow"

#     molecule_reactant = Molecule.from_options(
#         geom=[[x.symbol, *x.position] for x in atoms_reactant],
#         PES=pes_obj,
#         coord_obj=deloc_coords_reactant,
#         Form_Hessian=from_hessian
#     )

#     molecule_product = Molecule.copy_from_options(
#         molecule_reactant,
#         xyz=atoms_product.get_positions(),
#         new_node_id=num_nodes - 1,
#         copy_wavefunction=False
#     )

#     # optimizer
#     nifty.printcool("Building the Optimizer object")
#     opt_options = dict(print_level=1,
#                        Linesearch=line_search,
#                        update_hess_in_bg=not (only_climb or optimizer_method == "lbfgs"),
#                        conv_Ediff=conv_Ediff,
#                        conv_gmax=conv_gmax,
#                        DMAX=step_size_cap,
#                        opt_climb=only_climb)
#     if optimizer_method == "eigenvector_follow":
#         optimizer_object = eigenvector_follow.from_options(**opt_options)
#     elif optimizer_method == "lbfgs":
#         optimizer_object = lbfgs.from_options(**opt_options)
#     else:
#         raise NotImplementedError

#     # GSM
#     nifty.printcool("Building the GSM object")
#     gsm = DE_GSM.from_options(
#         reactant=molecule_reactant,
#         product=molecule_product,
#         nnodes=num_nodes,
#         CONV_TOL=conv_tol,
#         CONV_gmax=conv_gmax,
#         CONV_Ediff=conv_Ediff,
#         ADD_NODE_TOL=add_node_tol,
#         growth_direction=0,  # normal/react/prod: 0/1/2
#         optimizer=optimizer_object,
#         ID=ID,
#         print_level=1,
#         interp_method="DLC",
#     )

#     # optimize reactant and product if needed
#     if not fixed_reactant:
#         nifty.printcool("REACTANT GEOMETRY NOT FIXED!!! OPTIMIZING")
#         path = os.path.join(os.getcwd(), 'scratch', f"{ID:03}", "0")
#         optimizer_object.optimize(
#             molecule=molecule_reactant,
#             refE=molecule_reactant.energy,
#             opt_steps=100,
#             path=path
#         )
#     if not fixed_product:
#         nifty.printcool("PRODUCT GEOMETRY NOT FIXED!!! OPTIMIZING")
#         path = os.path.join(os.getcwd(), 'scratch', f"{ID:03}", str(num_nodes - 1))
#         optimizer_object.optimize(
#             molecule=molecule_product,
#             refE=molecule_product.energy,
#             opt_steps=100,
#             path=path
#         )

#     # set 'rtype' as in main one (???)
#     if only_climb:
#         rtype = 1
#     # elif no_climb:
#     #     rtype = 0
#     else:
#         rtype = 2

#     # do GSM
#     if restart_file:
#         nifty.printcool("Restarting GSM Calculation")
#         gsm.setup_from_geometries(geoms, reparametrize=reparametrize, start_climb_immediately=start_climb_immediately)
#     else:
#         nifty.printcool("Main GSM Calculation")

#     gsm.go_gsm(max_gsm_iterations, max_opt_steps, rtype=rtype)

#     # write the results into an extended xyz file
#     string_ase, ts_ase = gsm_to_ase_atoms(gsm)
#     write(f"opt_converged_{gsm.ID:03d}_ase.xyz", string_ase)
#     write(f'TSnode_{gsm.ID}.xyz', string_ase)

#     # post processing taken from the main wrapper, plots as well
#     post_processing(gsm, have_TS=True)

#     # cleanup
#     cleanup_scratch(gsm.ID)

# def gsm_to_ase_atoms(gsm: DE_GSM):
#     # string
#     frames = []
#     for energy, geom in zip(gsm.energies, gsm.geometries):
#         at = Atoms(symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom])
#         at.info["energy"] = energy
#         frames.append(at)

#     # TS
#     ts_geom = gsm.nodes[gsm.TSnode].geometry
#     ts_atoms = Atoms(symbols=[x[0] for x in ts_geom], positions=[x[1:4] for x in ts_geom])

#     return frames, ts_atoms

# def post_processing(gsm, analyze_ICs=False, have_TS=True):
#     plot(fx=gsm.energies, x=range(len(gsm.energies)), title=gsm.ID)

#     ICs = []
#     ICs.append(gsm.nodes[0].primitive_internal_coordinates)

#     # TS energy
#     if have_TS:
#         minnodeR = np.argmin(gsm.energies[:gsm.TSnode])
#         TSenergy = gsm.energies[gsm.TSnode] - gsm.energies[minnodeR]
#         print(" TS energy: %5.4f" % TSenergy)
#         print(" absolute energy TS node %5.4f" % gsm.nodes[gsm.TSnode].energy)
#         minnodeP = gsm.TSnode + np.argmin(gsm.energies[gsm.TSnode:])
#         print(" min reactant node: %i min product node %i TS node is %i" % (minnodeR, minnodeP, gsm.TSnode))

#         # ICs
#         ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
#         ICs.append(gsm.nodes[gsm.TSnode].primitive_internal_values)
#         ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
#         with open('IC_data_{:03d}.txt'.format(gsm.ID), 'w') as f:
#             f.write("Internals \t minnodeR: {} \t TSnode: {} \t minnodeP: {}\n".format(minnodeR, gsm.TSnode, minnodeP))
#             for x in zip(*ICs):
#                 f.write("{0}\t{1}\t{2}\t{3}\n".format(*x))

#     else:
#         minnodeR = 0
#         minnodeP = gsm.nR
#         print(" absolute energy end node %5.4f" % gsm.nodes[gsm.nR].energy)
#         print(" difference energy end node %5.4f" % gsm.nodes[gsm.nR].difference_energy)
#         # ICs
#         ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
#         ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
#         with open('IC_data_{}.txt'.format(gsm.ID), 'w') as f:
#             f.write("Internals \t Beginning: {} \t End: {}".format(minnodeR, gsm.TSnode, minnodeP))
#             for x in zip(*ICs):
#                 f.write("{0}\t{1}\t{2}\n".format(*x))

#     # Delta E
#     deltaE = gsm.energies[minnodeP] - gsm.energies[minnodeR]
#     print(" Delta E is %5.4f" % deltaE)

# def cleanup_scratch(ID):
#     cmd = "rm scratch/growth_iters_{:03d}_*.xyz".format(ID)
#     os.system(cmd)
#     cmd = "rm scratch/opt_iters_{:03d}_*.xyz".format(ID)
#     os.system(cmd)

# def print_msg():
#     msg = """
#     __        __   _                            _        
#     \ \      / /__| | ___ ___  _ __ ___   ___  | |_ ___  
#      \ \ /\ / / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \ 
#       \ V  V /  __/ | (_| (_) | | | | | |  __/ | || (_) |
#        \_/\_/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/ 
#                                     ____ ____  __  __ 
#                        _ __  _   _ / ___/ ___||  \/  |
#                       | '_ \| | | | |  _\___ \| |\/| |
#                       | |_) | |_| | |_| |___) | |  | |
#                       | .__/ \__, |\____|____/|_|  |_|
#                       |_|    |___/                    
# #==========================================================================#
# #| If this code has benefited your research, please support us by citing: |#
# #|                                                                        |# 
# #| Aldaz, C.; Kammeraad J. A.; Zimmerman P. M. "Discovery of conical      |#
# #| intersection mediated photochemistry with growing string methods",     |#
# #| Phys. Chem. Chem. Phys., 2018, 20, 27394                               |#
# #| http://dx.doi.org/10.1039/c8cp04703k                                   |#
# #|                                                                        |# 
# #| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
# #| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
# #| http://dx.doi.org/10.1063/1.4952956                                    |#
# #==========================================================================#


#     """
#     print(msg)

# def main():

#     # argument parsing and header
#     inpfileq = parse_arguments(verbose=True)
#     '''
#     # load calculators
#     if inpfileq["calc"].lower() == 'xtb':
#         from xtb.ase.calculator import XTB
#         calc = XTB(method="GFN2-xTB") # call GFN2-XTB
#     elif inpfileq["calc"].lower() == 'mace':
#         from mace.calculators import mace_off
#         calc = mace_off(model="medium", default_dtypes='float64') # call mace-off23
#     elif inpfileq["calc"].lower() == 'ani':
#         import torchani
#         calc = torchani.models.ANI1xnr(periodic_table_index=True).ase() # call ANI-1xnr
#     elif inpfileq["calc"].lower() == 'dpa2':
#         from deepmd.calculator import DP
#         calc = DP(model="/root/.cache/dpa2/dpa2-model.pt",head='Domains_Transition1x')
#     else:
#         from orb_models.forcefield import pretrained
#         from orb_models.forcefield.calculator import ORBCalculator
#         calc = ORBCalculator(pretrained.orb_v2(), device='cpu')
#     '''
#     # load input rxn
#     #input_rxn = read(inpfileq["xyzfile"], ":")
#     mols = xyz_parse(inpfileq["xyzfile"], multiple=True)
#     reactant = Atoms(symbols=mols[0][0], positions=mols[0][1])
#     product  = Atoms(symbols=mols[1][0], positions=mols[1][1])
    
#     wrapper_de_gsm(
#         reactant,
#         product,
#         inpfileq["calc"],
#         optimizer_method = inpfileq["optimizer"],
#         coordinate_type = inpfileq["coordinate_type"],
#         line_search = inpfileq["linesearch"],
#         step_size_cap = inpfileq["DMAX"],  # DMAX in the other wrapper
#         num_nodes = inpfileq["num_nodes"],  # 20 for SE-GSM
#         add_node_tol = inpfileq["ADD_NODE_TOL"],  # convergence for adding new nodes
#         conv_tol = inpfileq["CONV_TOL"],  # Convergence tolerance for optimizing nodes
#         conv_Ediff = inpfileq["conv_Ediff"],  # Energy difference convergence of optimization.
#         conv_gmax = inpfileq["conv_gmax"],  # Max grad rms threshold
#         ID = inpfileq["ID"],
#         nproc = inpfileq["nproc"],
#         max_gsm_iterations = inpfileq["max_gsm_iters"],
#         max_opt_steps = inpfileq["max_opt_steps"],  # 20 for SE-GSM
#         reparametrize = inpfileq["reparametrize"],
#         start_climb_immediately = inpfileq["start_climb_immediately"],
#         fixed_reactant = inpfileq["reactant_geom_fixed"],
#         fixed_product = inpfileq["product_geom_fixed"],
#         restart_file = inpfileq["restart_file"]
#         )

#     return
