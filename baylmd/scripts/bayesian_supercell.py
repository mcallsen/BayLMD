from mpi4py import MPI

from baylmd.settings import parse_settings
from baylmd.helpers import create_bins_supercell, get_displacements

from baylmd.workflows.forcefield_training import ForcefieldTraining

from ase.io import read, Trajectory

import ase.parallel as parallel

def main():
    start = 12830
    end = 12845

    settings = parse_settings("settings.yaml")
    settings.internal.calculate_phi = True

    # Read the structure and the trajectory.
    structure = read(settings.forcefield.structure_file)
    supercell = read(settings.supercell.structure_file)

    # Get the scaling factor between the structure and the supercell.
    factor = len(supercell) / len(structure)

    # Get a list of atom indices for each copy of structure that fits into the supercell. If supercell is not an
    # integer multiple of structure, then the last entry in the list will be a copy of structure containing
    # the remaining fractional part of the supercell.
    indices = create_bins_supercell(supercell, structure, axis = 2)

    trajectory = Trajectory(settings.dynamics.trajectory_file)

    # Setup the force-field.
    training = ForcefieldTraining(settings)
    training.start()

    current = structure.copy()
    for iteration in range(start, end):
        errors = list()
        positions = trajectory[iteration].get_scaled_positions()

        # convert scaled positions of the supercell to the base cell.
        positions[:, 2] *= factor
        for layer in indices:
            # set the positions according to the current copy of the structure. Note that for a non-integer
            # supercell the periodic boundary conditions will lead to a uniform translation of the
            # unit-cell. This will cause problems without ASRs.
            current_positions = positions[layer]
            current.set_scaled_positions(current_positions)

            # Get the center of the current copy of structure.
            center = current_positions[:, 2].mean()

            # Compute the Bayesian error for this copy of structure
            displacements = get_displacements(structure, current.get_scaled_positions())
            phi = training.model.phi(displacements)

            if parallel.world.rank == 0:
                bayesian_error = training.estimator.score(phi.toarray())
                errors.append((bayesian_error, center))
            parallel.world.barrier()
            

        parallel.parprint(f"Iteration: {iteration}")
        if parallel.world.rank == 0:      
            for index, error in enumerate(errors):
                print(f"    {index}  {error[0]:.5g}  {error[1]:.3f}")
        parallel.world.barrier()

if __name__ == "__main__": 
    main()

