from ase import Atoms
from numpy.typing import ArrayLike

import numpy as np

def parse_structure(file_name: str, format: str = "vasp") -> Atoms:
    """ Parse a structure from VASP POSCAR. """
    from ase.io import read
    return read(file_name, format=format)

def parse_fit_data(file_name: str) -> tuple[str, ArrayLike]:
    """ Parse positions or forces from a file in XDATCAR format."""
    with open(file_name, "r") as fo:
        lines = fo.readlines()
        mode = lines[0].split()[0]
        natoms = int(lines[0].split()[1])
        data = np.empty((0, natoms, 3), dtype=float)

        i = 0
        while(i < len(lines)):
            i += 1
            tmp = np.zeros((1, natoms, 3), dtype=float)
            for j in range(natoms):
                tmp[0, j] = type_cast(lines[i].split(), float)
                i += 1
            data = np.vstack((data, tmp))
            # i += 1
        return mode, data

def parse_bayesian_error(file_name: str, column: int = 2):
    return np.loadtxt(file_name, usecols=column)

def parse_tag(line: str, tag: str, func):
    key, value = line.split(":")
    if key.strip() != tag:
        print(f"PARSING ERROR: looking for '{tag}' but found '{key}'.")
    return type_cast(value, func)

def parse_vector_tag(line: str, tag: str, func):
    key, value = line.split(":")
    if key.strip() != tag:
        print(f"PARSING ERROR: looking for '{tag}' but found '{key}'.")
    return type_cast(value.split(), func)

def type_cast(item, func):
    """ Convert a list of items to a different type. """
    if isinstance(item, list):
        return list(map(func, item))
    return func(item)
