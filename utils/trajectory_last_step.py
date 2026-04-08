from ase.io import Trajectory

import sys

trajectory_file = sys.argv[1]
file_name = sys.argv[2]

index = -1

trajectory = Trajectory(trajectory_file)
new_trajectory = Trajectory(file_name, mode="w")
new_trajectory.write(trajectory[index])
