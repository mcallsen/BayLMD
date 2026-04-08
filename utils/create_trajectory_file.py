from ase.io import read, Trajectory
  
from forcefield_ml.settings import parse_settings

import os
import sys

mode = sys.argv[1]

settings = parse_settings("settings.yaml")

structure_file = settings.forcefield.structure_file
if mode == "supercell":
    structure_file = settings.supercell.structure_file
if mode == "file":
    structure_file = sys.argv[2]

print(f"Reading structure from: {structure_file}")

structure = read(structure_file)

print(len(structure))

if not os.path.isfile(settings.dynamics.trajectory_file):
    with Trajectory(settings.dynamics.trajectory_file, mode="w") as trajectory:
        trajectory.write(structure)
        trajectory.close()
