from forcefield_ml.helpers import create_bins

import numpy as np
import ase.units as units

def make_layers(atoms_per_layer, number):
    layers = list()
    for i in range(number):
        layers.append(list(range((i * atoms_per_layer), (i + 1) * atoms_per_layer)))
    return layers

def sine_profile(number):
    x = np.linspace(0, 1, number, endpoint=False)
    return np.sin(x * 2 * np.pi)

def get_temperatures(atoms):
    factor = 2 / (3 * units.kB) 
    momenta = atoms.get_momenta()
    velocities = atoms.get_velocities()
    temperatures = [0.5 * factor * np.dot(v, w) for v, w in zip(momenta, velocities)]
    return np.array(temperatures)

def get_temperature_profile(atoms, axis = 0):
    x = atoms.get_scaled_positions(wrap = False)[:, axis]
    y = get_temperatures(atoms)
    return x, y

def print_temperature_profile(supercell, atoms, number):
    layers = create_bins(supercell, number)
    temperatures = get_temperatures(atoms)
    delta = 1.0 / number
    for i, indices in enumerate(layers):
         x = (i + 0.5) * delta
         print(x, temperatures[indices].mean())
