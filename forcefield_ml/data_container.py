from __future__ import annotations

from numpy.typing import ArrayLike

from forcefield_ml.parsing import type_cast

import numpy

class DataContainer:
    """ A Container holding the training data. """
    def __init__(self, n: int) -> None:
        self.displacements: ArrayLike = numpy.empty((0, n))
        self.forces: ArrayLike = numpy.empty((0, n))

    def __len__(self) -> int:
        return len(self.displacements)

    @classmethod 
    def read(cls, file_name: str) -> DataContainer:
        """ Read a structure container from a file. """
        with open(file_name, 'r') as fo:
            count, dimension = type_cast(fo.readline().split()[1:], int)
            container = cls(dimension)

            if count > 0:
                data = numpy.loadtxt(file_name).reshape((-1, dimension, 2))

                container.displacements = data[:count, :, 0]
                container.forces = data[:count, :, 1]

            return container

    @property
    def shape(self):
        return self.displacements.shape

    def write(self, file_name: str) -> None:
        """ Write the data container to a file. """
        count, dimension = self.displacements.shape
        with open(file_name, "w") as fo:
            fo.write(f"# {count} {dimension}\n")
            for i in range(count):
                for j in range(dimension):
                    fo.write(f"{self.displacements[i, j]} {self.forces[i, j]}\n")

    def add_dataset(self, displacements: ArrayLike, forces: ArrayLike) -> None:
        """ Add a structure and the corresponding forces to the container. """
        self.displacements = numpy.vstack((self.displacements, displacements))
        self.forces = numpy.vstack((self.forces, forces))

    def distance(self, positions: ArrayLike) -> float:
        """ Get a measure for the abstract distance of the structure w.r.t. the data in this container. """
        return 1.0

    @property
    def data(self) -> tuple[ArrayLike, ArrayLike]:
        return (self.displacements, self.forces)

