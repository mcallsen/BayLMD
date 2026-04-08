from __future__ import annotations

from typing import Tuple, Type
from numpy.typing import ArrayLike

import numpy as np

class Metrics:
    def __init__(self) -> None:
        self.mean: float = 0.0
        self.interquartile_range: Tuple[float, float] = (0.0, 0.0)
        self._rmses: list = []

    @property
    def rmses(self):
        """ Get the rsmes as numpy array. """
        return np.array(self._rmses)

    @classmethod
    def from_data(cls: Type, predictions: ArrayLike, targets: ArrayLike) -> Metrics:
        """ Initialise a Metrics object based on arrays of predictions and targets. """
        metrics: Metrics = cls()
        for prediction, target in zip(predictions, targets):
            metrics.add_dataset(prediction, target)
        metrics.update()
        return metrics

    def add_dataset(self, prediction: ArrayLike, target: ArrayLike) -> None:
        """ Add a tuple of prediction and target to the Metrics. """
        self._rmses.append(self.calculate_rmse(prediction, target))

    def update(self) -> None:
        """ Update the mean value and the interquartile range. """
        # Calculate the mean value.
        rmses = self.rmses
        self.mean = rmses.mean()

        # Calculate the interquartile range.
        rmses = np.sort(rmses)
        count_half = int(len(rmses) / 2)
        self.interquartile_range = (np.median(rmses[:count_half]), np.median(rmses[count_half:]))

    @staticmethod
    def calculate_rmse(predictions: ArrayLike, targets: ArrayLike) -> float:
        """ Calculate the RMSE for one pair of predictions, targets. """
        return np.sqrt(((predictions - targets) ** 2).mean())

    @staticmethod
    def calculate_r2(predictions: ArrayLike, targets: ArrayLike) -> float:
        """ Calculate the R2 for a pair of predictions and targets. """
        res = ((predictions - targets) ** 2).mean()
        tot = ((targets - np.mean(targets)) ** 2).mean()
        return 1 - res / tot