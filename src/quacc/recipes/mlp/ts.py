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

from typing import TYPE_CHECKING

from ase.mep.neb import NEB
import rmsd

from quacc import job
from quacc.recipes.mlp._base import pick_calculator
from quacc.runners.ase import Runner
from quacc.schemas.ase import Summarize
from quacc.schemas.ase import summarize_neb_run
from quacc.utils.dicts import recursive_dict_merge

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

    if alignment_method == "kabsch":
        product_atoms.set_positions(rmsd.kabsch_rotate(product_atoms.get_positions(), reactant_atoms.get_positions()))

    if interpolation_method == "geodesic":
        images = geodesic_interpolate_wrapper(
            relax_summary_r["atoms"], relax_summary_p["atoms"], **interpolate_flag
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

    dyn = Runner(images, calc).run_neb(neb_flags, opt_flags, run_flags)

    # return Summarize(
    #     additional_fields={"name": f"{method} NEB"} | (neb_kwargs or {})
    # ).run(dyn)

    return {
        "initial_images": images,
        "neb_results": summarize_neb_run(
            dyn,
            n_images=len(images),
            additional_fields={
                "name": f"{method} NEB",
                "method": method,
                "neb_flags": neb_flags,
                "opt_flags": opt_flags,
                "run_flags": run_flags,
            },
        ),
    }