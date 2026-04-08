from ase.io import Trajectory, read

from forcefield_ml.helpers import combine_structures
from forcefield_ml.settings import parse_settings

import sys

left_file = sys.argv[1]
right_file = sys.argv[2]
output_file = sys.argv[3]

index = -1

settings = parse_settings("settings.yaml")
supercell = read(settings.supercell.structure_file, format="vasp")

# Read the two trajectories
left = Trajectory(left_file)
right = Trajectory(right_file)

combined = combine_structures(supercell, left[index], right[index], axis = 2)

trajectory = Trajectory(output_file, mode="w")
trajectory.write(combined)
