from ase import Atoms

from mpi4py import MPI

#from baylmd import parallel

from baylmd.context import Context, to_string_bayesian, to_string_dynamics
from baylmd.helpers import create_bins
from baylmd.windows import RollingWindow
from baylmd.parsing import parse_structure
from baylmd.settings import ParametersEnum, parse_settings, Settings

from baylmd.workflows.molecular_dynamics import MolecularDynamics
from baylmd.workflows.forcefield_training import ForcefieldTraining

import ase.units as units

import numpy as np

def run_dynamics(dynamics: MolecularDynamics, context: Context) -> None:

    # constants required for calculating the temperature on both sides of a larger supercell.
    factor = 2.0 / (3 * len(dynamics.equilibrium_structure) * units.kB)
    left, right = create_bins(dynamics.equilibrium_structure, number = 2, axis = 2)

    for _ in range(dynamics.iteration, dynamics.settings.max_iterations):
        # Perform one step of MD.
        dynamics.update(context)
        
        # Update the temperature on both sides.
        momenta = dynamics.atoms.get_momenta()
        velocities = dynamics.atoms.get_velocities()

        context.temperature_left = factor * np.vdot(momenta[left], velocities[left])
        context.temperature_right = factor * np.vdot(momenta[right], velocities[right])
        context.delta_t = context.temperature_left - context.temperature_right

        dynamics.print_context(context, to_string_dynamics)

def run_bayesian(dynamics: MolecularDynamics, context: Context, settings: Settings) -> None:
    average = RollingWindow(settings.active_learning.max_size)
    for _ in range(dynamics.iteration, dynamics.settings.max_iterations):
        # Perform one step of MD.
        dynamics.update(context)
        
        context.bayesian_error = dynamics.bayesian_error()
        average.update(context.bayesian_error)
        context.mean_error = average.mean

        dynamics.print_context(context, to_string_bayesian)

def main():
    # TODO: Update the molecular dyncamics to the new architecture i.e. outsource the the 
    # MD part to an independent executable for mpirun.

    # Parse settings from an input file.
    settings = parse_settings("settings.yaml")

#   parallel.initialize_groups(1)

    if settings.training.parameters_mode == ParametersEnum.TRAIN:
        # Setup the forcefield for the training cell.
        training = ForcefieldTraining(settings)
        training.start()

    if settings.supercell is not None:
        settings.supercell.description = "Supercell"
        settings.forcefield = settings.supercell

    # Finally setup the molecular dynamics workflow.
    structure: Atoms = parse_structure(settings.forcefield.structure_file)

    dynamics = MolecularDynamics(structure, settings)

    context = Context()

    if settings.internal.calculate_phi:
        run_bayesian(dynamics, context, settings)
    else:
        run_dynamics(dynamics, context)


if __name__ == "__main__":
    main()
