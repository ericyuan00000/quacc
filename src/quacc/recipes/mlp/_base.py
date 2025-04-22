"""Base functions for universal machine-learned interatomic potentials."""

from __future__ import annotations

from functools import lru_cache
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    from ase import Atoms
    from ase.calculators.calculator import Calculator

LOGGER = getLogger(__name__)


@lru_cache
def pick_calculator(
    method: Literal["mace-mp-0", "mace-off", "m3gnet", "chgnet", "newtonnet", "escaip"], **kwargs
) -> Calculator:
    """
    Adapted from `matcalc.util.get_universal_calculator`.

    Parameters
    ----------
    method
        Name of the calculator to use
    **kwargs
        Custom kwargs for the underlying calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely. For a list of available
        keys, refer to the `mace.calculators.mace_mp`, `chgnet.model.dynamics.CHGNetCalculator`,
        or `matgl.ext.ase.M3GNetCalculator` calculators.

    Returns
    -------
    Calculator
        The chosen calculator
    """
    import torch

    if not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available to PyTorch. Calculations will be slow.")

    
    if method.lower() == 'ani':
        from torchani.calculator import ANICalculator
        from torchani import __version__
        calc = ANICalculator(**kwargs)
    # elif method.lower() == 'chg':
    #     from chgnet.model.dynamics import chgnet_finetuned, CHGNetCalculator
    #     model = chgnet_finetuned(device=device)
    #     sae_path    = weight_path.replace('ts1x-tuned.ckpt','chgnet_ts1x_atom_ref.npy')
    #     model       = chgnet_finetuned(weight_path,sae_path,device=device)
    #     calculator  = CHGNetCalculator(model, device=device)
    elif method.lower() == 'left':
        from oa_reactdiff.trainer.calculator import LeftNetCalculator
        from oa_reactdiff import __version__
        calc = LeftNetCalculator(**kwargs)
    elif method.lower() == 'mace':
        from mace.calculators import mace_off, mace_off_finetuned
        from mace import __version__
        calc = mace_off_finetuned(**kwargs)
    elif method.lower() == 'orb':
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        from orb_models import __version__
        weights_path = kwargs.pop("weights_path", None)
        model = pretrained.orb_v2_finetuned(weights_path)
        calc = ORBCalculator(model, **kwargs)


    elif method.lower() == "m3gnet":
        import matgl
        from matgl import __version__
        from matgl.ext.ase import PESCalculator

        model = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
        kwargs.setdefault("stress_weight", 1.0 / 160.21766208)
        calc = PESCalculator(potential=model, **kwargs)

    elif method.lower() == "chgnet":
        from chgnet import __version__
        from chgnet.model.dynamics import CHGNetCalculator

        calc = CHGNetCalculator(**kwargs)

    elif method.lower() == "mace-mp-0":
        from mace import __version__
        from mace.calculators import mace_mp

        if "default_dtype" not in kwargs:
            kwargs["default_dtype"] = "float64"
        calc = mace_mp(**kwargs)

    elif method.lower() == "mace-off" or method.lower() == "mace":
        from mace import __version__
        from mace.calculators import mace_off

        if "default_dtype" not in kwargs:
            kwargs["default_dtype"] = "float64"
        model_path = kwargs.pop("model_path", None)
        calc = mace_off(**kwargs)

    elif method.lower() == "newtonnet":
        from newtonnet import __version__
        from newtonnet.utils.ase_interface import MLAseCalculator

        calc = MLAseCalculator(**kwargs)

    elif method.lower() == "escaip":
        # import yaml
        # import numpy as np
        # from torch import nn
        # from torch_geometric.data import Data
        # from ase.calculators.calculator import Calculator
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        from fairchem.core import __version__
        import sys
        # sys.path.append('/global/homes/e/ericyuan/GitHub')
        sys.path.append('/global/homes/e/ericyuan/GitHub/EScAIP')
        # from EScAIP.src import EfficientlyScaledAttentionInteratomicPotential

        # class EScAIP(nn.Module):
        #     def __init__(self, config_file, checkpoint_file):
        #         super().__init__()
        #         with open(config_file) as f:
        #             config = yaml.safe_load(f)
        #         checkpoint = torch.load(checkpoint_file, map_location='cuda')
        #         self.module = EfficientlyScaledAttentionInteratomicPotential(**config['model']).to('cuda')
        #         self.load_state_dict(checkpoint['state_dict'])
        #         self.normalizers = checkpoint['normalizers']
        #         self.eval()

        #     def forward(self, data):
        #         output = self.module(data)
        #         for key in output.keys():
        #             output[key] = output[key] * self.normalizers[key]['std'] + self.normalizers[key]['mean']
        #         return output
            
        #     def ase_data(self, atoms: Atoms) -> Data:
        #         return Data(
        #             atomic_numbers=torch.tensor(atoms.numbers),
        #             pos=torch.tensor(atoms.positions).float(),
        #             cell=torch.tensor(np.array(atoms.cell)).float(),
        #             batch=torch.tensor([0 for _ in range(len(atoms))], dtype=torch.long),
        #             natoms=torch.tensor([len(atoms)]),
        #             num_graphs=1,
        #         )
            
        #     def ase_data_batch(self, atoms_list: list[Atoms]) -> Data:
        #         return Data(
        #             atomic_numbers=torch.cat([torch.tensor(atoms.numbers) for atoms in atoms_list]), 
        #             pos=torch.cat([torch.tensor(atoms.positions).float() for atoms in atoms_list]), 
        #             cell=torch.cat([torch.tensor(np.array(atoms.cell)).float() for atoms in atoms_list]), 
        #             batch=torch.tensor([i for i, atoms in enumerate(atoms_list) for _ in range(len(atoms))], dtype=torch.long),
        #             natoms=torch.tensor([len(atoms) for atoms in atoms_list]),
        #             num_graphs=len(atoms_list),
        #         )

        # class EScAIPCalculator(Calculator):
        #     implemented_properties = ["energy", "forces"]

        #     def __init__(self, config_file, checkpoint_file, **kwargs):
        #         super().__init__(**kwargs)
        #         self.model = EScAIP(config_file, checkpoint_file)

        #     def calculate(self, atoms=None, properties=None, system_changes=None, **kwargs):
        #         super().calculate(atoms, properties, system_changes)
        #         data = self.model.ase_data(atoms)
        #         output = self.model(data)
        #         self.results["energy"] = output["energy"].item()
        #         self.results["forces"] = output["forces"].detach().cpu().numpy()

        # calc = EScAIPCalculator(**kwargs)

        calc = OCPCalculator(**kwargs)

    else:
        raise ValueError(f"Unrecognized {method=}.")

    calc.parameters["version"] = __version__

    return calc
