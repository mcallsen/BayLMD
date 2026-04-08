from typing import List

import numpy as np


def load_forces(keys: List[str]):
    forces = [np.loadtxt(f"forces_{key}.dat") for key in keys]
    return np.sum(forces, axis=0)

def main():
    contributions = {
        "anharmonicity": [2, 3, 4],
        "2nd order": [2],
        "3rd order": [3],
        "4th order": [4]
    }

    forces_full = np.loadtxt(f"forces_total.dat")
    value_full = np.square(forces_full).mean(axis=0).sum()

    for contribution, keys in contributions.items():
        forces = load_forces(keys)
        value = np.square(forces).mean(axis=0).sum()
        print(f"{contribution}:", np.sqrt(value, value_full))

if __name__ == "__main__": 
    main()