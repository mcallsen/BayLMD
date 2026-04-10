from ase import Atoms
from typing import List

from forcefield_ml.context import Context
from forcefield_ml.helpers import randomize_structure, count_lines
from forcefield_ml.parsing import parse_structure
from forcefield_ml.settings import parse_settings, DynamicsEnum

from forcefield_ml.workflows.abinitio_calculation import AbinitioCalculation
#from forcefieldml.workflows.active_learning import ActiveLearning
#from forcefieldml.workflows.molecular_dynamics import MolecularDynamics
from forcefield_ml.workflows.forcefield_training import ForcefieldTraining

import subprocess

ACTIVE_LEARNING_SCRIPT = "~/forcefieldml/forcefieldml/scripts/run_active_learning.py"

def update_training_data(structures: List[Atoms], calculation: AbinitioCalculation, training: ForcefieldTraining) -> None:
    context = Context()

    for structure in structures:
        # Calculate the ab initio forces for this structure
        context.structure = structure

        calculation.update(context)

        # Add the current structure to the training set.
        training.add_dataset(structure, context.forces)


def main():
    # Parse arguments. Eventually this should be read from the command line or from a file.
    settings = parse_settings("settings.yaml")
    settings.internal.parallel = False

    command = f"mpirun -np {settings.abinitio.mpi_procs} python3 -m mpi4py {ACTIVE_LEARNING_SCRIPT}"
 
    # Print the settings for the non persistent workflows. 
    #ActiveLearning.print_settings(settings)
    AbinitioCalculation.print_settings(settings)
    #MolecularDynamics.print_settings(settings)

    # Read the input structure.
    equilibrium_structure: Atoms = parse_structure(settings.forcefield.structure_file)

    # Create the ActiveLearning workflow
    calculation = AbinitioCalculation(settings)
    training = ForcefieldTraining(settings)

    training.model.print_settings(settings)
    training.print_settings(settings)

    # Perform the ab initio calculations for the initial structures, if required.
    #if settings.dynamics.dynamics_mode is not DynamicsEnum.CONTINUE:
        # starting from scratch. Get the structures required for the training.
    if settings.active_learning.initial_iterations > 0:
        structures = []
        if settings.dynamics.dynamics_mode is DynamicsEnum.FROM_FILE:
            # TODO: This requires reading the trajectory.
            # Structures are read from a file.
            # structures = self.molecular_dynamics.dynamics.trajectory[:self.settings.initial_iterations]
            pass
        else:
            # Create N randomised structures.
            structures = [randomize_structure(equilibrium_structure) for _ in range(settings.active_learning.initial_iterations)]

        # Get the corresponding training data.
        update_training_data(structures, calculation, training)

    training.start()

    context = Context()
    while context.iteration < settings.dynamics.max_iterations:
        # Run the Active learning program until a new structure is found.
        subprocess.run(command, shell=True, check=True)

        # Update the current iteration number.
        context.iteration = count_lines(settings.dynamics.log_file + "_0") - 1

        # Check whether a refit is required.
        if context.iteration < settings.dynamics.max_iterations:
            # Read the structures from the _tmp directory
            structures = [parse_structure(f"./_tmp/POSCAR_{index}") for index in range(1)]
            update_training_data(structures, calculation, training)
            training.update(context)


if __name__ == "__main__":
    main()
