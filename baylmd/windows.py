from queue import Queue

import numpy as np

class RollingWindow:
    """ A rolling mean value and variance based on Pythons builtin FIFO Queue. """
    def __init__(self, max_size: int = 0, initial_value: float = 0.0) -> None:
        self.values = Queue(max_size)
        self.deltas = Queue(max_size)

        self._mean: float = 0.0
        self._sum_squares: float = 0.0

        if initial_value > 0:
            self.update(initial_value)

    @property
    def mean(self) -> float:
        """ The current rolling average. """
        return self._mean

    @property
    def variance(self) -> float:
        """ The current rolling variance. """
        return self._sum_squares / self.values.qsize()

    @property
    def standard_deviation(self) -> float:
        """ The current standard deviation. """
        return np.sqrt(self.variance)

    def update(self, value: float) -> None:
        """ Update the mean value iteratively. """
        if self.values.full():
            # First remove an old value.
            self._replace(value)
            return
        self._add(value)

    def _add(self, value: float) -> None:
        """ Incrementally add a new value to the mean. """
        self.values.put(value)

        previous_mean = self._mean

        # update the mean value.
        self._mean += (value - previous_mean) / self.values.qsize()

        # update the sum of squares according to Welford's online algorithm.
        delta = (value - previous_mean) * (value - self.mean)
        self._sum_squares += delta
        self.deltas.put(delta)

    def _replace(self, value: float) -> None:
        """ Update the mean by adding a new value and removing the oldest. """
        # Remove old values from the queues.
        previous_value = self.values.get()
        previous_delta = self.deltas.get()
        
        self.values.put(value)

        previous_mean = self._mean
        self._mean += (value - previous_value) / self.values.qsize()

        delta = (value - previous_mean) * (value - self.mean)
        self._sum_squares += delta - previous_delta

        self.deltas.put(delta)