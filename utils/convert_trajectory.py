from ase.io import Trajectory, write

import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

trajetory = Trajectory(input_file)
write(output_file, trajetory, format="vasp-xdatcar")
