from forcefield_ml.helpers import create_bins
from forcefield_ml.settings import parse_settings
from ase.io import read, Trajectory

import sys

def main():

    index = int(sys.argv[1])
    layer = int(sys.argv[2])

    settings = parse_settings("settings.yaml")

    # Read the structure and the trajectory.
    structure = read(settings.forcefield.structure_file)
    supercell = read(settings.supercell.structure_file)

    number = int(len(supercell) / len(structure))

    indices = create_bins(supercell, number, axis = 2)

    trajectory = Trajectory(settings.dynamics.trajectory_file)

    positions = trajectory[index].get_scaled_positions()
    positions[:, 2] *= number

    current = structure.copy()
    current.set_scaled_positions(positions[indices[layer]])

    current.write("test.poscar")

if __name__ == "__main__": 
    main()
