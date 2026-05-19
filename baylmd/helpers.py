from ase import Atoms
from numpy.typing import ArrayLike
from typing import List, Tuple, Optional

from baylmd.parsing import parse_fit_data, parse_structure

import shutil
import numpy as np
import math
import os

TOLERANCE: float = 0.000001

def transform(matrix: ArrayLike, positions: ArrayLike) -> ArrayLike:
    """Apply a transformation to a matrix of vectors."""
    return np.matmul(matrix, positions.transpose()).transpose()

def wrap(vector: ArrayLike, lower: float, upper: float) -> None:
    """ Wrap a vector to a given interval."""
    for index in range(len(vector)):
        if vector[index] < lower: vector[index] += 1.0
        if vector[index] > upper: vector[index] -= 1.0

def get_displacements(structure: Atoms, positions: ArrayLike, subtract_average: bool = False) -> ArrayLike:
    """ Calculate the displacements for a single structure."""
    scaled = structure.get_scaled_positions()
    differences = positions - scaled
    for index in range(len(differences)):
        wrap(differences[index], -0.5, 0.5)
    
    # Convert the displacements to cartesian coordinates.
    differences = transform(structure.cell.transpose(), differences)

    # subtract the mean value of the displacements if required.
    if subtract_average:
        differences = subtract_mean_displacements(differences)
    return differences.flatten()

def subtract_mean_displacements(displacements: ArrayLike) -> ArrayLike:
    """ Subtract the average from an array of displacements. """
    average = displacements.mean(axis=0)
    return displacements - average

def prepare_displacements(structure: Atoms, structures: ArrayLike) -> ArrayLike:
    """ Calculate the displacements from the equilibrium structure for a set of positions."""
    displacements = list()
    for index in range(len(structures)):
        displacements.append(get_displacements(structure, structures[index]))
    return np.array(displacements)

def get_structures_and_forces(structure_file: str, positions_file: str, force_file: str) -> tuple[Atoms, ArrayLike, ArrayLike]:
    """ Get the structure, displacements and forces from files."""
    structure = parse_structure(structure_file)
    natoms = len(structure)
    structures = np.loadtxt(positions_file).reshape((-1, natoms, 3))
    forces = np.loadtxt(force_file).reshape((-1, natoms, 3))
    return structure, prepare_displacements(structure, structures), forces

def get_fit_data(model, displacements: ArrayLike, forces: ArrayLike, indices: list[int], scale: float = 1.0) -> tuple[ArrayLike, ArrayLike]:
    """ Get the X and y matrices for the machine learning from a ForceConstantModel."""
    from scipy.sparse import csr_matrix, vstack
    X = csr_matrix((0, model.n_parameters), dtype=float)
    y = np.empty(0, dtype=float)
    rows = model.rows
    columns = model.columns
    for index in indices:
        data = model.GetFitMatrix(displacements[index] / scale)
        X = vstack([X, csr_matrix((data, (rows, columns)), dtype=float)])
        y = np.hstack((y, forces[index].flatten()))
    return X, y

def multivariate_gaussian(t, mu, sigma) -> float:
    det = np.linalg.det(2 * np.math.pi * sigma)
    _sigma = np.linalg.inv(sigma)
    vector = t - mu
    exponent = - 0.5 * (np.math.ln(det) + np.dot(vector, np.dot(_sigma, vector)))
    return np.math.exp(exponent)

def copy_files(source: str, destination: str, files: List[str], suffix: str = "") -> None:
    """ Copy a list of files from the source directory to the destination directory. """
    from os import path
    for file in files:
        file_name = file + suffix
        shutil.copyfile(path.join(source, file), path.join(destination, file_name))

def count_files(directory: str, pattern: str) -> int:
    """ Count the number of files matching pattern in directory. """
    from glob import glob
    return len(glob(directory + "/" + pattern))

def count_lines(file_name: str) -> int:
    """ Count the number of lines in a file. """
    with open(file_name) as fo:
        return sum(1 for _ in fo)

def initialize_file(file_name: str, default_content: str="") -> Optional[str]:
    """ Initialize a file with 'default_content' if it does not exist. """
    if not os.path.isfile(file_name):
        with open(file_name, "w") as fo:
            fo.write(default_content)
        return f"File '{file_name}' does not exists. Creating with default content."
    return None

def make_directories(directories: List[str]) -> None:
    """ Make a couple of directories in the current workdir. """
    from os import makedirs, getcwd, path
    cwd = getcwd()
    for directory in directories:
        new_directory = path.join(cwd, directory)
        if not path.exists(new_directory):
            makedirs(new_directory)

def randomize_structure(structure: Atoms, width: float = 0.01) -> Atoms:
    rng = np.random.default_rng()
    number_of_atoms = structure.get_global_number_of_atoms()
    displacements = width * rng.standard_normal(number_of_atoms * 3)
    random_structure = structure.copy()
    random_structure.positions += displacements.reshape((number_of_atoms, 3))
    return random_structure

def get_indices_in_interval(positions: ArrayLike, lower: float, upper: float, axis: int, epsilon: float) -> List:
    """ Get the indices of all atoms in the interval [lower, upper). """
    return [index for index, position in enumerate(positions) if position[axis] >= lower - epsilon and position[axis] < upper - epsilon]

def create_bins(atoms: Atoms, number: int, axis: int = 2, epsilon: float = TOLERANCE, lower: float = 0.0, upper: float = 1.0) -> List:
    """ Subdivide a section [lower, upper] of a supercell in 'number' slices along 'axis'. """
    x = np.linspace(lower, upper, number + 1)
    positions = atoms.get_scaled_positions(wrap=False)
    indices = list()
    for i in range(number):
        indices.append(get_indices_in_interval(positions, x[i], x[i + 1], axis=axis, epsilon=epsilon))
    return indices

def create_bins_supercell(supercell: Atoms, structure: Atoms, axis: int = 2, epsilon: float = TOLERANCE) -> List:
    """ 
    Find a list of copies of structure that covers supercell.
    
    If supercell is not an integer multiple of structure, add another copy of structure starting from the 
    other end of the supercell that includes the fractional bit.
    """
    factor = len(supercell) / len(structure)
    fractional, number = math.modf(factor)
    width = 1.0 / factor

    indices = create_bins(supercell, int(number), axis=axis, epsilon=epsilon, upper=number * width)
    if fractional > 0.0:
        indices.append(get_indices_in_interval(supercell.get_scaled_positions(wrap=False), 1.0 - width, 1.0, axis=axis, epsilon=epsilon))
    return indices

def get_shifted_positions(structure: Atoms, axis: int = 0, number: int = 2, displacement: float = 0.0) -> ArrayLike:
    positions = structure.get_scaled_positions(wrap = False)
    positions[:, axis] /= number
    positions[:, axis] += displacement
    return positions

def combine_structures(supercell: Atoms, structure_left: Atoms, structure_right: Atoms, axis: int = 0) -> Atoms:
    # Take a copy of the supercell structure
    structure = supercell.copy()

    # Get the atom indices for the left and the right side of the supercell.
    left, right = create_bins(supercell, number = 2, axis = axis)

    # Get the shifted positions for the left and the right hand side.
    positions_left = get_shifted_positions(structure_left, axis = axis)
    positions_right = get_shifted_positions(structure_right, axis = axis, displacement = 0.5)

    # combine the positions into one array.
    positions = supercell.get_scaled_positions()


    print(len(structure_left), len(left), len(right))

    #print(supercell.get_center_of_mass(scaled = True, indices = left), structure_left.get_center_of_mass(scaled = True))
    #print(supercell.get_center_of_mass(scaled = True, indices = right), structure_right.get_center_of_mass(scaled = True))

    positions[left] = positions_left
    positions[right] = positions_right

    # combine the momenta into one array.
    momenta = supercell.get_momenta()
    momenta[left] = structure_left.get_momenta()
    momenta[right] = structure_right.get_momenta()

    # set the positions and momenta of the new structure.
    structure.set_scaled_positions(positions)
    structure.set_momenta(momenta)

    return structure
