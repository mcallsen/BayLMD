from typing import Dict, List
from abc import ABC, abstractmethod
from enum import Enum
from scipy.sparse import csc_matrix, vstack
from numpy.typing import ArrayLike

from baylmd.timing import timing

import ase.parallel as parallel
import numpy


class Orbit(object):
    """ Helper class defining an Orbit. """
    def __init__(self, order: int = 0, index: int = -1, prefactor: float = 0, count: int = 0, radius: float = 0.0):
        self.order: int = order
        self.index: int = index
        self.prefactor: float = prefactor
        self.count: int = count
        self.radius: float = radius

    @property
    def indices(self) -> range:
        return range(self.index, self.index + 3**self.order)


class SplitMode(Enum):
    ORDER = "order"
    RADIUS = "radius"
    ALL = "all"


class TensorIndices(object):
    def __init__(self, orbit: Orbit) -> None:
        self.order: int = orbit.order
        self.radius: float = orbit.radius
        self.indices: List[int] = list(orbit.indices)

    def key(self, split_mode: SplitMode):
        if split_mode == SplitMode.ORDER:
            return f"{self.order}"
        return f"{self.order}  {self.radius:.3f}"

    def extend(self, indices: List[int]) -> None:
        self.indices.extend(indices)

    def get_nonzero(self, values: ArrayLike):
        result = list()

        indices = numpy.nonzero(values)[0]
        for index in indices:
            if index in self.indices:
                result.append(values[index])
        return result

class Forcefield(ABC):
    """ Forcefield is an abstraction layer over the underlying Force Model."""
    def __init__(self, file_name: str) -> None:
        super().__init__()
        self.file_name: str = file_name
        self.description: str = ""
        self.current: csc_matrix = None
        self.model = None
        self.cutoffs: List[float] = list()
        self.orbits: List[Orbit] = list()

    @property
    @abstractmethod
    def n_components(self) -> int:
        """ Property for the number of force components (3 * number of atoms). """

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """ Property for the total number of independent parameters. """

    @timing
    def get_fitdata(self, displacements: ArrayLike, forces: ArrayLike, indices: List[int] = None) -> tuple[csc_matrix, csc_matrix]:
        if indices is None:
            indices = range(len(displacements))

        x = csc_matrix((0, self.model.n_parameters), dtype=float)
        y = numpy.empty(0, dtype=float)

        for index in indices:
            x = vstack([x, self.phi(displacements[index])])
            y = numpy.hstack((y, forces[index]))

        return x, y

    def set_parameters(self, parameters: ArrayLike) -> None:
        self.model.parameters = parameters

    @abstractmethod
    def get_forces(self, displacements: ArrayLike) -> ArrayLike:
        """ Get the forces corresponding to these displacements. """

    @abstractmethod
    def phi(self, displacements: ArrayLike) -> csc_matrix:
        """ Get the phi matrix for these displacements. """

    @abstractmethod
    def write(self, file_name: str) -> None:
        """ Write the Model to a file. """

    def get_tensor_indices(self, split_mode = SplitMode.ORDER) -> List[TensorIndices]:
        tensor_indices: List[TensorIndices] = list()
        for orbit in self.orbits:
            tensor_indices.append(TensorIndices(orbit))

        # If we want all orbits separately we are done here.
        if split_mode == SplitMode.ALL:
            return tensor_indices

        # combine the tensors according to the split mode.
        tensor_dict: Dict[str, TensorIndices] = dict()
        for tensor in tensor_indices:
            key = tensor.key(split_mode)
            if key in tensor_dict:
                tensor_dict[key].extend(tensor.indices)
                continue
            tensor_dict[key] = tensor

        return list(tensor_dict.values())
