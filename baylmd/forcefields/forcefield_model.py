from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix

from baylmd.forcefield import Forcefield, Orbit
from baylmd.parsing import parse_tag, parse_vector_tag
from baylmd.settings import InternalSettings, Settings, ForcefieldSettings

from baylmd.forceModelPy import ForceModel

import ase.parallel as parallel

import numpy 

class ForcefieldModel(Forcefield):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.forcefield.forcefield_file)

        # Store the relevant settings.
        self.forcefield: ForcefieldSettings = settings.forcefield
        self.internal: InternalSettings = settings.internal

        self.model = ForceModel(self.forcefield.forcefield_file, self.internal.parallel, settings.internal.mpi_group)

        #self._read_forcefield_file(self.forcefield.forcefield_file)

        self.parameters = None

    @property
    def n_components(self) -> int:
        return self.model.n_components

    @property
    def n_parameters(self) -> int:
        return self.model.n_parameters

    def set_parameters(self, parameters: ArrayLike) -> None:
        if len(parameters) != self.n_parameters:
            # Number of parameters is not the same. This can be correct, so just print a warning.
            if parallel.world.rank == 0:
                print(f"Warning: set_parameters expected {self.n_parameters} parameters, but got {len(parameters)}.")

        self.parameters = parameters
        self.model.parameters = parameters
        #parallel.parprint(*numpy.nonzero(parameters)[0])

    def phi(self, displacements: ArrayLike) -> csc_matrix:
        # The forcefield model will return the matrix as a flattened 1d array so we need to reshape.
        return csc_matrix(numpy.array(self.model.get_fit_matrix(displacements)).reshape((len(displacements), -1)))

    def write(self, file_name: str) -> None:
        if file_name is None:
            file_name = self.file_name
        self.model.Write(file_name)

    def get_forces(self, displacements: ArrayLike) -> ArrayLike:
        if self.internal.calculate_phi:
            # calculate phi explicitly, which is required for Bayesian error estimation but also 
            # requires communicating phi in the MPI case.
            self.current = self.phi(displacements)
            return self.current.dot(self.parameters)

        # Use the faster way, where only the force components are communicated.
        self.forces = numpy.array(self.model.get_forces(displacements))
        return self.forces

    def _read_forcefield_file(self, file_name: str) -> None:
        """ Read information from the forcefield file."""
        #if parallel.world.rank == 0:
        with open(file_name) as file_object:
            lines = file_object.readlines()
            current = 0

            # Parse the header.
            self.description = parse_tag(lines[current], "description", str)
            self.cutoffs = parse_vector_tag(lines[current + 1], "cutoffs", float)

            norbits = parse_tag(lines[current + 5], "orbits", int)

            # Advance the pointer to the first orbit.
            current += 6

            # Parse the orbits.
            for _ in range(norbits):
                # Parse the current orbit.
                order = parse_tag(lines[current + 1], "order", int)
                index = parse_tag(lines[current + 2], "tensor_index", int)
                prefactor = parse_tag(lines[current + 3], "prefactor", float)
                radius = parse_tag(lines[current + 4], "radius", float)

                self.orbits.append(Orbit(order=order, index=index, prefactor=prefactor, radius=radius))

                # Advance the pointer to the next Orbit.
                nclusters = parse_tag(lines[current + 5], "clusters", int)
                current += 6 + nclusters
        parallel.world.barrier()

    def print_settings(self, settings: Settings) -> None:
        print(f"\n{self.__class__.__name__}\n")
        print("    Forcefield file:         ", settings.forcefield.forcefield_file)
        print("    Structure file:          ", settings.forcefield.structure_file)
        print("    Description:             ", self.description)
        print("    Cutoffs:                 ", self.cutoffs)
        print("    Number of atoms:         ", int(self.n_components / 3))
        print("    Number of parameters:    ", self.n_parameters)
        print("    Parallel:                ", self.internal.parallel)
        print("    Calculate phi:           ", self.internal.calculate_phi)
        print("")
