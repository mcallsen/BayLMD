from ase.io import read
from forcefield_ml.helpers import get_structures_and_forces
from forcefield_ml.data_container import DataContainer

import sys

count = int(sys.argv[1])
natoms = int(sys.argv[2])

# Collect all the structures
with open("structures.dat", "w") as fo:
    for i in range(1, count + 1):
        structure = read(f"output/POSCAR_{i}", format="vasp")
        fo.write(f"Direct {natoms} {i}\n")
        for position in structure.get_scaled_positions():
            fo.write(f"{position[0]} {position[1]} {position[2]}\n")

_, displacements, forces = get_structures_and_forces("POSCAR.vasp", "structures.dat", "forces.dat")

container = DataContainer(3 * natoms)
for d, f in zip(displacements, forces):
    container.add_dataset(d, f.flatten())
container.write("ml_training_data.dat")
