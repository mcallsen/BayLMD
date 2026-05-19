from abc import ABC, abstractmethod
from ase import Atoms
from typing import List
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, vstack


from csld.symmetry_structure import SymmetrizedStructure
from csld.structure import SupercellStructure
from csld.lattice_dynamics import init_ld_model, LDModel
from csld.phonon.phonon import Phonon

from baylmd.forcefield import Forcefield, Orbit
from baylmd.helpers import wrap, TOLERANCE
from baylmd.parsing import parse_structure
from baylmd.settings import Settings

import numpy


class TranslationMap(object):
    """ Helper class for mapping atom indices uppon translation. """
    def __init__(self, translation: List[int], size: int) -> None:
        self.translation: List[int] = translation

        # Check whether this is the trivial translation map.
        is_zero = all(x == 0 for x in translation)
        self.index_map: ArrayLike = numpy.arange(size) if is_zero else numpy.full(size, -1)

    def map(self, indices: List[int]) -> List[int]:
        """ Map a list of indices. """
        return [self.index_map[index] for index in indices]

    def contains(self, index: int) -> bool:
        """ Check whether an index is already known"""
        return self.index_map[index] != -1


class Supercell(ABC):
    """ Representation of a Supercell to translate clusters from primitive to supercell. """
    def __init__(self, supercell_file: str, primitive_structure) -> None:
        self.primitive_structure = primitive_structure
        self.supercell: Atoms = parse_structure(supercell_file)
        self.supercell_file: str = supercell_file
        self.natoms: int = len(self.supercell)   

    @abstractmethod
    def translate_cluster(self, cluster):
        """ Return a list with all translated copies of the cluster """
        

class IntegerMultipleSupercell(Supercell):
    """ 
    Supercell class for integer multiples of the primitive cell. 

    Periodic boundary conditions for integer multiples of the primitive cell allow 
    for a very efficient mapping of the atoms to the supercell.
    """
    def __init__(self, supercell_file: str, primitive_structure):
        super().__init__(supercell_file, primitive_structure)     

        # Find the supercell indices
        B = numpy.linalg.inv(primitive_structure.lattice_vectors())
        indices = numpy.diagonal(numpy.matmul(self.supercell.cell, B))
        self.indices = numpy.round(indices).astype(int)

        # If there is an odd numver basis vectors exchanged between the primitive and the supercell
        # the supercell indices can become negative.
        self.indices = numpy.abs(self.indices)

        self.count = numpy.prod(self.indices)

        self.translations = numpy.array([[x, y, z] for z in range(self.indices[2]) for y in range(self.indices[1]) for x in range(self.indices[0])])

        # Create the ijkl map
        self.ijkl_map = numpy.zeros((*self.indices, primitive_structure.frac_coords.shape[0]), dtype=int)
        for index, position in enumerate(self.supercell.get_scaled_positions()):
            ijkl = primitive_structure.frac2ijkl(position * self.indices, tolerance=0.00001)
            ijkl[:3] = numpy.mod(ijkl[:3], self.indices)
            self.ijkl_map[tuple(ijkl)] = index

    def translate_cluster(self, cluster):
        return [self._translate_cluster(cluster, translation) for translation in self.translations]

    def _translate_cluster(self, cluster, translation):
        indices = []
        for ijkl in cluster.ijkls:
            l = ijkl[3]
            ijk = numpy.mod(numpy.array(ijkl[:3]) + translation, self.indices)
            indices.append(self.ijkl_map[(*ijk, l)])
        return indices


class GeneralSupercell(Supercell):
    """ 
    Supercell class for non-integer multiples of the primitive cell. 

    Because the the supercell is not an integer multiple of the primitive cell, we cannot apply
    periodic boundary conditions of the primitive cell when mapping atoms. 
    """
    def __init__(self, supercell_file: str, primitive_structure):
        super().__init__(supercell_file, primitive_structure)        

        # Find the conversion matrix between primitive and supercell.
        cell_inv = numpy.linalg.inv(primitive_structure.lattice.matrix)
        self.sc_matrix = numpy.matmul(cell_inv, self.supercell.cell.transpose())
        self.sc_matrix_inv = numpy.linalg.inv(self.sc_matrix)

        # Find the ijkl indices for each atom in the supercell and store them as a map index -> ijkl.
        ijkls = list()
        translations = list()
        for position in self.supercell.get_scaled_positions():
            # Get the ijkl indices in terms of the primitive cell.
            ijkl = primitive_structure.frac2ijkl(self.convert_to_primitive(position), tolerance=TOLERANCE)
            ijkls.append(ijkl)
            if ijkl[-1] == 0:
                # This is another copy of the first atom in the primitive cell, which means we
                # found a new translation vector.
                translations.append(ijkl[:3])

        self.count: int = len(translations)

        # Create all the translation maps
        self.translations: List[TranslationMap] = [TranslationMap(translation, self.natoms) for translation in translations]

    def translate_cluster(self, cluster) -> List[List[int]]:
        """ Returns a list with all translated copies of a cluster within this supercell. """
        clusters = list()

        # Find the atom indices for this cluster in the supercell.
        indices = self.find_cluster_indices(cluster)

        # Apply all translations to the cluster.
        for translation in self.translations:
            # Check whether all the indices have been found previously. In principle, this should
            # not be necessary after the 2nd order terms, because by then all atoms should have been
            # looked at at least once.
            for i, index in enumerate(indices):
                if not translation.contains(index):
                    # Find the mapped index and update the translation map.
                    ijkl = cluster.ijkls[i].copy()
                    ijkl[:3] += translation.translation
                    translation.index_map[index] = self.find_atom_index(ijkl)

            # Map the cluster and add it to the list.
            clusters.append(translation.map(indices))
        return clusters

    def find_cluster_indices(self, cluster) -> List[int]:
        """ Find the atom indices for a given cluster in the supercell. """
        return [self.find_atom_index(ijkl) for ijkl in cluster.ijkls]

    def find_atom_index(self, indices: List[int], tolerance: float = TOLERANCE) -> int:
        """ Find the atom index in the supercell for [i, j, k, l] in the primitive cell. """
        # get the fractional coordinates w.r.t. the primitive cell.
        position = self.primitive_structure.ijkl2frac(indices)

        # Convert the coordinates to the supercell and apply PBC.
        position = self.convert_to_supercell(position)
        wrap(position, 0.0, 1.0)

        # Find an atom with these coordinates in the supercell.
        for index, other in enumerate(self.supercell.get_scaled_positions()):
            difference = position - other
            wrap(difference, -0.5, 0.5)
            if numpy.linalg.norm(difference) <= tolerance:
                return index

        # Did not find any atom at this position.
        print(f"Error: Atom index not found! {indices}")
        return -1

    def convert_to_supercell(self, position: ArrayLike) -> ArrayLike:
        return self.sc_matrix_inv.dot(position)

    def convert_to_primitive(self, position: ArrayLike) -> ArrayLike:
        return self.sc_matrix.dot(position)


class CSLDSupercell(Supercell):
    """ Supercell class that uses the original CSLD implementation. """
    def __init__(self, supercell_file: str, primitive_structure: SymmetrizedStructure):
        super().__init__(supercell_file, primitive_structure)

        self.structure: SupercellStructure = SupercellStructure.from_file(primitive_structure, supercell_file)
        self.count: int = len(self.structure.ijk_ref)

    def translate_cluster(self, cluster, translation):
        return super().translate_cluster(cluster, translation)
    

class CsldModel(Forcefield):
    def __init__(self, settings: Settings, accoustic_sum_rules: bool = True, supercell_class: type = CSLDSupercell) -> None:
        super().__init__(settings.forcefield.forcefield_file)

        self.primitive_structure = SymmetrizedStructure.init_structure(settings.training.csld_settings, 2, False)
        self.model: LDModel = init_ld_model(self.primitive_structure, settings.training.csld_settings, {}, 2, 2, 2)

        self.supercell: Supercell = None
        self.initialize_supercell(settings.forcefield.structure_file, supercell_class)

        self.cmat: csr_matrix = self.model.Cmat.T if accoustic_sum_rules else self.model.Cmat1.T
        self.parameters = None
        self.settings = settings.training.csld_settings

        self._set_orbits()

    @property
    def n_components(self) -> int:
        """ Property for the number of force components (3 * number of atoms). """
        return 3 * self.supercell.natoms

    @property
    def n_parameters(self) -> int:
        """ Property for the total number of independent parameters. """
        return self.cmat.shape[1]

    def phi(self, displacements: ArrayLike) -> csr_matrix:
        amat = self.model.calc_correlation(displacements.reshape((-1, 3)), self.all_clusters)
        return amat.dot(self.cmat)[:-1]

    def get_fitdata(self, displacements: ArrayLike, forces: ArrayLike) -> tuple[csr_matrix, csr_matrix]:
        x = csr_matrix((0, self.n_parameters), dtype=float)
        y = numpy.empty(0, dtype=float)

        for disp, force in zip(displacements, forces):

            x = vstack((x, self.phi(disp)))
            y = numpy.hstack((y, force))
        return x, y
        
    def set_parameters(self, parameters: ArrayLike) -> None:
        self.parameters = parameters

    def get_forces(self, displacements: ArrayLike) -> ArrayLike:
        """ Get the forces corresponding to these displacements. """
        self.current = self.phi(displacements)
        return self.current.dot(self.parameters)

    def initialize_supercell(self, supercell_file: str, supercell_class: type = CSLDSupercell):
        self.supercell = supercell_class(supercell_file, self.primitive_structure)
        self._set_orbits()

    def translate_to_supercell(self, cluster):
        if isinstance(self.supercell, CSLDSupercell):
            return self.model.translate_cluster_to_supercell(self.supercell.structure, cluster)
        return self.supercell.translate_cluster(cluster)

    def write(self, file_name: str) -> None:
        # The c_matrix in CSLD contains empty rows and columns for 0th and 1st order FCTs. Compute the
        # corresponding row and column offsets.
        row_offset = 0
        column_offset = 0
        offset = 0

        # create the orbits.
        for orbit in self.model.orbits:
            if orbit.order > 1:
                break
            row_offset += orbit.ncorr_full
            offset += 1

        for order in self.model.fct_ord:
            if order < 2:
                column_offset += 1
            else:
                break

        # Get the rotation matrices in cartesian coordinates.
        rotations = numpy.array([op.rot_inv for op in self.model.prim.spacegroup]).flatten()

        # The first seven rows are 0th and 1st order tensors, which we are ignoring.
        c_matrix = self.cmat.toarray()[row_offset:, column_offset:]

        with open(file_name, "w") as fo:
            # Write the header with some basic information about the force-field.
            fo.write(f"description: {self.settings['description']}\n")
            fo.write(f"cutoffs: {self.settings['cluster_diameter']}\n")
            fo.write(f"components: {3 * self.supercell.natoms}\n")
            fo.write(f"parameters_total: {c_matrix.shape[0]}\n")
            fo.write(f"parameters_unique: {c_matrix.shape[1]}\n")

            # Write the orbits.
            fo.write(f"orbits: {len(self.orbits)}\n")
            for i, orbit in enumerate(self.orbits):
                fo.write(f"orbit {i}:\n")
                fo.write(f"    order:        {orbit.order}\n")
                fo.write(f"    tensor_index: {orbit.index}\n")
                fo.write(f"    prefactor:    {orbit.prefactor}\n")
                fo.write(f"    radius:       {orbit.radius}\n")
                fo.write(f"    clusters:     {orbit.count}\n")

                # Loop through all the clusters in the orbit, translate them to the supercell and write them to the file.
                csld_orbit = self.model.orbits[i + offset]
                for ig, primitive_cluster in zip(csld_orbit.clusters_ig, csld_orbit.clusters):
                    clusters = self.translate_to_supercell(primitive_cluster)
                    for cluster in clusters:
                        fo.write(f"{ig} " + " ".join([str(x) for x in cluster]) + "\n")

            # Write the rotation matrices.
            fo.write(f"Rotations: {len(rotations)}\n")
            fo.write(f'{" ".join(str(int(i)) for i in rotations)}\n')

            # Write the C-matrix in sparse format.
            fo.write(f"Tensors: {c_matrix.shape[0]} {c_matrix.shape[1]}\n")
            for index, row in enumerate(c_matrix):
                columns = numpy.nonzero(row)[0]
                fo.write(f"{index} {len(columns)}\n")
                for column in columns:
                    fo.write(f"{column}: {row[column]}\n")

    def save_force_constants(self, parameters: ArrayLike, structure_file: str = None, supercell_file: str = None, order: int = 2) -> None:
        """ 
        Use the CSLD interface to write the FORCE_CONSTANTS. 
        
        Because the CSLD interface does not work for non-integer supercells (e.g. a supercell based on the
        conventional cubic cell) this function explicitely requires the file names for the 'base' and the 
        'super' cell.
        """
        
        # Read the base structure and initialise the Phonon object.
        if structure_file is None:
            structure_file = self.settings['prim']
        structure: SymmetrizedStructure = SymmetrizedStructure.init_structure({'prim': structure_file}, 2, False, check_prim = 0)
        phonon = Phonon(structure, self.model, parameters, False)

        # This requires a CSLD supercell w.r.t. the 'base' structure (not necessarily the primitive cell).
        if supercell_file is None:
            supercell_file = self.supercell.supercell_file
        supercell: SupercellStructure = SupercellStructure.from_file(structure, supercell_file)

        for i in range(2, order + 1):
            if i == 2:
                phonon.export_hessian_forshengbte(supercell)
            if i in [3, 4]:
                self.model.save_fcshengbte(parameters, i)

    def _set_orbits(self) -> None:
        self.orbits = []
        index = 0 
        for orbit in self.model.orbits:
            if orbit.order < 2:
                continue
            self.orbits.append(Orbit(order=orbit.order, index=index, prefactor=orbit.cluster.factorial, count=self.supercell.count*len(orbit.clusters), radius=orbit.cluster.diameter))
            index += orbit.ncorr_full

