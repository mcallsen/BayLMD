from ase import Atoms
from numpy.typing import ArrayLike

from dataclasses import dataclass

@dataclass
class Context:
    structure: Atoms = None
    displacements: ArrayLike = None
    forces: ArrayLike = None
    parameters: ArrayLike = None
    sigma: ArrayLike = None
    bayesian_error: float = 0
    mean_error: float = 0
    vasp_criterion: float = 0
    temperature: float = 0
    temperature_left: float = 0
    temperature_right: float = 0
    delta_t: float = 0
    total_energy: float = 0
    iteration: int = 0
    last_refit: int = -5
    refit_required: bool = False


def to_string_bayesian(context: Context):
    postfix = "training" if context.refit_required else ""
    return f"{context.iteration:>10}  {context.total_energy:>10.3f}  {context.bayesian_error:>5.2e}  {context.mean_error:>5.2e}  {context.temperature:>7.1f}    {postfix} \n"

def to_string_dynamics(context: Context):
    return f"{context.iteration:>10}  {context.total_energy:>10.3f}  {context.temperature_left:>7.1f}  {context.temperature_right:>7.1f}  {context.delta_t:>7.1f}  {context.temperature:>7.1f}\n"
