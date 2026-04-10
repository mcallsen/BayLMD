from typing import Tuple
from mpi4py import MPI
from numpy.typing import ArrayLike

from forcefield_ml.forcefield import SplitMode, Forcefield
from forcefield_ml.settings import parse_settings

from forcefield_ml.workflows.forcefield_training import ForcefieldTraining

import ase.parallel as parallel
import numpy

def write_to_file(file_name: str, forces, iteration: int = 0):
    n_atoms = len(forces)
    with parallel.paropen(file_name, 'a') as fo:
        fo.write(f"Direct {n_atoms} {iteration}\n")
        for vector in forces:
            fo.write(f"{vector[0]} {vector[1]} {vector[2]}\n")
        fo.write("\n")

def calculate_forces(forcefield: Forcefield, displacements: ArrayLike, file_name: str = None, indices = None):
    if indices is not None:
        forcefield.model.set_parameter_mask(indices)
    forces = numpy.array([forcefield.get_forces(disp) for disp in displacements])
    if file_name is not None:
        numpy.savetxt(file_name, forces)
    return forces

def calculate_force_contibution(forcefield: Forcefield, displacements: ArrayLike, file_name: str = None, indices = None) -> Tuple[float, ArrayLike]:
    forces = calculate_forces(forcefield, displacements, file_name, indices)
    value = numpy.square(forces).mean(axis=0).sum()
    norms = [numpy.linalg.norm(vector, axis=1) for vector in forces.reshape((len(displacements), -1, 3))]
    return value, numpy.array(norms)

def main():
    split_mode = SplitMode.RADIUS
    settings = parse_settings("settings.yaml")

    # Setup the force-field.
    training = ForcefieldTraining(settings)
    training.start()

    parallel.world.barrier()

    parameters = numpy.loadtxt(settings.training.parameters_file)
    training.model.set_parameters(parameters)

    forcefield = training.model
    displacements = training.training_data.displacements

    # Get the indices of the parameters for each contibution.
    tensors = forcefield.get_tensor_indices(split_mode)

    parallel.parprint("Got the tensors.")

    # Get the values for the total force.
    value_full, norms_full = calculate_force_contibution(forcefield, displacements)

    parallel.parprint("Got the full values.")

    # write the individual force contributions to files.
    for tensor in tensors:
        value, norms = calculate_force_contibution(forcefield, displacements, indices = tensor.indices)

        # comparing width of distributions of forces, similar to the anharmonicity measure.
        x = numpy.sqrt(value / value_full)

        # dividing each force by the total force and then taking the average.
        y = numpy.sqrt((norms / norms_full).mean())

        parallel.parprint(tensor.key(split_mode), f"{x:.5f}  {y:.5f}")

if __name__ == "__main__": 
    main()
