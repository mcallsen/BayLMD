from ase import Atoms
from typing import List
from numpy.typing import ArrayLike

from ase.calculators.calculator import Calculator, all_changes

from baylmd.forcefield import Forcefield
from baylmd.helpers import get_displacements

import numpy as np

import ase.parallel as parallel

class ForcefieldCalculator(Calculator):
    """ An ASE calculator using a Forcefield object to compute energy and forces. """
    implemented_properties = ['energy', 'forces'] # ASE special magic to define implemented properties.

    def __init__(self, structure: Atoms, model: Forcefield, atoms: Atoms = None) -> None:
        super().__init__(atoms=atoms)
        self.equilibrium_structure: Atoms = structure
        self.model: Forcefield = model
        self.displacements: ArrayLike = np.zeros(3 * len(structure))

    def calculate(self, atoms: Atoms = None, properties: List[str] = ['energy, forces'], system_changes: List[str] = all_changes) -> None:
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        # Get the displacements of the atoms.
        self.displacements = self.get_displacements(self.atoms)

        # Get the forces acting on the atoms.
        forces: ArrayLike = self.model.get_forces(self.displacements)

        # Compute the energy from the forces.
        energy: float = np.dot(forces, self.displacements)

        # Attach the energy and forces to results.
        self.results['forces'] = forces.reshape((len(self.equilibrium_structure), 3))
        self.results['energy'] = energy

    def get_displacements(self, atoms: Atoms) -> ArrayLike:
        """ Get the displacements of atoms w.r.t. the equilibrium structure. """
        return get_displacements(self.equilibrium_structure, atoms.get_scaled_positions())
