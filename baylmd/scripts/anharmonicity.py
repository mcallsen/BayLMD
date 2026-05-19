from mpi4py import MPI

from baylmd.forcefield import SplitMode
from baylmd.settings import parse_settings
from baylmd.helpers import get_displacements
from baylmd.forcefields import ForcefieldModel

from baylmd.scripts.force_contribution import calculate_forces
from baylmd.workflows.forcefield_training import ForcefieldTraining

from ase.io import read

import ase.parallel as parallel
import numpy

def main():
    split_mode = SplitMode.ORDER
    settings = parse_settings("settings.yaml")

    # Setup the force-field.
    training = ForcefieldTraining(settings)
    training.start()

    if settings.supercell is not None:
        settings.supercell.description = "Supercell"
        settings.forcefield = settings.supercell

    forcefield = ForcefieldModel(settings)

    parameters = numpy.loadtxt(settings.training.parameters_file)
    forcefield.set_parameters(parameters)

    # Get the displacements.
    start = 25000
    end = 75000

    structure = read(settings.forcefield.structure_file)
    trajectory = read(settings.dynamics.trajectory_file, format="vasp-xdatcar", index=":")

    # Print the contributions to the forces from each order.
    displacements = list()
    for atoms in trajectory[start:end]:
        displacements.append(get_displacements(structure, atoms.get_scaled_positions()))

    # Read the forcefield file to get the orbit information.
    # TODO: Quick fix, this should eventually be done in forcefield.get_tensor_indices()
    forcefield._read_forcefield_file(forcefield.forcefield.forcefield_file)

    # Get the indices of the parameters for each contibution.
    tensors = forcefield.get_tensor_indices(split_mode)

    # Get the total force.
    forces_total = calculate_forces(forcefield, displacements, "forces_total.xdatcar")
    value_full = numpy.square(forces_total).mean(axis=0).sum()


    # compute anharmonicity
    forces_harmonic = calculate_forces(forcefield, displacements, indices = tensors[0].indices)
    value_anharmonic = numpy.square(forces_total - forces_harmonic).mean(axis=0).sum()
    parallel.parprint("anharmonicity:", numpy.sqrt(value_anharmonic / value_full))

    # write the individual force contributions to files.
    for tensor in tensors:
        key = tensor.key(split_mode)
        forces = calculate_forces(forcefield, displacements, f"forces_{key}.xdatcar", indices = tensor.indices)
        value = numpy.square(forces).mean(axis=0).sum()
        parallel.parprint(key, numpy.sqrt(value / value_full))

if __name__ == "__main__": 
    main()
