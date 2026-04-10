from __future__ import annotations

from numpy.typing import ArrayLike

import numpy as np

class GaussianEstimator:
    def __init__(self, ncoefficients: int) -> None:
        # Design matrices and targets.
        self.phis: ArrayLike = None
        self.targets: ArrayLike = None

        # dimensions.
        self.ncoefficients: int = ncoefficients

        # internal variables.
        self.mean: ArrayLike = None
        self.S: ArrayLike = None
        self._sigma: ArrayLike = None
        self.lambdas: ArrayLike = None

        self.alpha: float = 1.0
        self.beta: float = 1.0
        self.gamma: float = 0.0

    @property
    def parameters(self):
        return np.real(self.mean)

    @classmethod
    def initialise_and_fit(cls, phis: ArrayLike, targets: ArrayLike) -> GaussianEstimator:
        """ Initialise a GaussianEstimator based on phis and target values. """
        estimator: GaussianEstimator = cls(phis.shape[-1])
        estimator.fit(phis, targets)
        return estimator

    @classmethod
    def from_file(cls, file_name: str) -> GaussianEstimator:
        """ Initialise a GaussianEstimator from a file containing the S matrix. """
        s = np.loadtxt(file_name)
        estimator: GaussianEstimator = cls(s.shape[0])
        estimator.S = s
        return estimator

    def get_sigma(self, phi: ArrayLike) -> ArrayLike:
        """ Get the Sigma matrix for this phi. """
        return np.matmul(phi, np.matmul(self.S, phi.transpose()))

    def fit(self, phis: ArrayLike, targets: ArrayLike) -> None:
        """ Perform Ridge regression. """
        self.phis = phis
        self.targets = targets

        # The mean and the standard deviation for the parameter distribution.
        self.mean = np.zeros(self.ncoefficients) 
        self.S = np.zeros((self.ncoefficients, self.ncoefficients))

        # Compute Phis^T * Phis as well as it's eigenvalues. This has to be done only once. 
        self._sigma = np.matmul(phis.transpose(), phis)
        self.lambdas = np.linalg.eig(self._sigma)[0]

        self._iterate_parameters()

    def score(self, phi) -> float:
        """ Compute the supremum norm of the bayesian error. """
        return self.get_sigma(phi).max()

    def write(self, file_name: str) -> None:
        """ Write the S matrix to a file. """
        np.savetxt(file_name, np.real(self.S))

    def _iterate_parameters(self, tolerance: float = 0.01) -> None:
        """ Find optimal values for alpha and beta by maximising the evidence function. """
        converged = False
        while not converged:
            _alpha, _beta = self.alpha, self.beta
            self._iteration_step()
            if (abs(_alpha - self.alpha) < tolerance and abs(_beta - self.beta) < tolerance):
                converged = True

        # Update the mean and sigma with the converged alpha and beta.
        self._update_coefficients()

    def _update_coefficients(self) -> None:
        """ Update the coefficients of the Ridge regression with the current alpha and beta. """
        S = self.alpha * np.eye(self.ncoefficients) + self.beta * self._sigma
        self.S = np.linalg.inv(S)
        self.mean = self.beta * np.dot(self.S, np.dot(self.phis.transpose(), self.targets))

    def _iteration_step(self) -> None:
        """ Perform a single iteration step to find alpha and beta. """
        self._update_coefficients()

        self.gamma = 0
        for l in self.lambdas:
            self.gamma += l / ( self.alpha / self.beta + l)

        self.alpha = self.gamma / (np.dot(self.mean, self.mean))
        beta = 0
        for p, t in zip(self.phis, self.targets):
            beta += np.linalg.norm(t - np.dot(p, self.mean)) ** 2
        self.beta = (self.phis.shape[0] - self.gamma) / beta



