""" Workflows for training forcefields. """
from typing import Optional, Type
from numpy.typing import ArrayLike
from ase import Atoms

from forcefield_ml.context import Context
from forcefield_ml.data_container import DataContainer
from forcefield_ml.estimator import GaussianEstimator
from forcefield_ml.forcefields import ForcefieldModel
from forcefield_ml.helpers import initialize_file, get_displacements
from forcefield_ml.metrics import Metrics
from forcefield_ml.parsing import parse_structure
from forcefield_ml.settings import Settings, TrainingSettings
from forcefield_ml.workflow import Workflow

from os import getcwd

# There are two libraries implementing glmnet in Python. The first one is dicontinued but 
# appears to be more reliable than the successor. So for now we will continue using glment.

#from glmnet import ElasticNet
from scipy.sparse import csr_matrix, vstack

#from glmnet_python.cvglmnet import cvglmnet
#from glmnet_python.cvglmnetCoef import cvglmnetCoef
#from scipy.sparse import csc_matrix, vstack

import numpy as np
import ase.parallel as parallel


class ForcefieldTraining(Workflow):
    """ A Workflow for training a forcefield. """
    def __init__(self, settings: Settings,  parent: Optional[Workflow] = None, model: ForcefieldModel = None, forcefield_class: Type = ForcefieldModel) -> None:
        super().__init__(parent=parent)

        self.model = None
        if model is not None:
            self.model = model
        else:
            self.model = forcefield_class(settings)

        self.settings: TrainingSettings = settings.training
        self.calculate_phi: bool = settings.internal.calculate_phi

        # Read the input structure.
        self.equilibrium_structure: Atoms = parse_structure(settings.forcefield.structure_file)

        # Read the training data.
        message = initialize_file(self.settings.data_file, default_content=f"# 0 {self.model.n_components}")
        if message is not None:
            parallel.parprint(f"[{self.__class__.__name__}]", message)

        self.training_data = DataContainer.read(self.settings.data_file)

        #self.lasso = ElasticNet(**self.settings.lasso_keywords)
        #self._fit: dict = dict()

        self.estimator: GaussianEstimator = None
        self.metrics = Metrics()

        self._parameters = np.zeros(self.model.n_parameters)

        self.X = csr_matrix((0, self.model.n_parameters), dtype=float)
        #self.X = csc_matrix((0, self.model.n_parameters), dtype=float)

        self.y = np.empty(0, dtype=float)

        self.iteration: int = 1

    @property
    def parameters(self) -> ArrayLike:
        return self._parameters

    def start(self) -> None:
        # Train the forcefield or read the parameters from a file.
        if len(self.training_data) > 0:
            for index in range(len(self.training_data)):
                self._add_fit_data(self.training_data.displacements[index], self.training_data.forces[index])
            self._train_forcefield()

    def update(self, context: Context) -> None:
        # Update should be called once the training data has changed.
        self._train_forcefield()

    def add_dataset(self, structure: Atoms, forces: ArrayLike):
        """ Add a new set of displacements and forces to the training data. """
        displacements = get_displacements(self.equilibrium_structure, structure.get_scaled_positions())
        self._add_fit_data(displacements, forces)

        # Update the training data and write the new training data to file.
        self.training_data.add_dataset(displacements, forces)
        self.write(self.settings.data_file)

    def write(self, file_name: str) -> None:
        """ Write the training data to a file. """
        self.training_data.write(getcwd() + "/" + file_name)

    def print_settings(self, settings: Settings) -> None:
        print(f"\n{self.__class__.__name__}\n")
        print("    Parameters file:        ", settings.training.parameters_file)
        print("    Training data file:     ", settings.training.data_file)
        print("    Training mode:          ", settings.training.parameters_mode.value)
        print("    Training structures:    ", len(self.training_data))
        print("")

    def _print_training(self):
        #index = np.where(self.lasso.lambda_path_ == self.lasso.lambda_best_)[0][0]

        print(f"Training (iteration {self.iteration}):")
        print("")
        print("    Number of structures:   ", len(self.training_data))
        print("    Non-zero parameters:    ", len(np.nonzero(self.parameters)[0]))
        print("    Gamma:                  ", f"{self.estimator.gamma:.3f}")
        print("    alpha:                  ", f"{self.estimator.alpha:.5f}")
        print("    sqrt(1/beta):           ", f"{np.sqrt(1/self.estimator.beta):.5f}")
        #print("    CV mean score:          ", self.lasso.cv_mean_score_[index])
        #print("    Lambda:                 ", self.lasso.lambda_best_[0])
        print("")

        #lambda_best = self._fit["lambda_1se"][0]
        #index = np.where(self._fit["lambdau"] == lambda_best)[0][0]

        #print(f"Training (iteration {self.iteration}):")
        #print("")
        #print("    Non-zero parameters:    ", self._fit["nzero"][index])
        #print("    R^2 score:              ", self._fit["glmnet_fit"]["dev"][index])
        #print("    Lambda:                 ", lambda_best)
        #print("")

    def _add_fit_data(self, displacements: ArrayLike, forces: ArrayLike):
        phi = self.model.phi(displacements)
        if parallel.world.rank == 0:
            self.X = vstack([self.X, phi])
            self.y = np.hstack((self.y, forces))

    def _train_forcefield(self) -> None:
        """ Train the forcefield using an ElasticNet. """
        if parallel.world.rank == 0:
            # Fit the LASSO using GLMNET
            #self.lasso = ElasticNet(**self.settings.lasso_keywords)
            #self.lasso.fit(self.X, self.y)
            #self._fit = cvglmnet(self.X, self.y, **self.settings.glmnet_keywords)

            # Setup the Gaussian estimator.
            self.estimator = GaussianEstimator.initialise_and_fit(self.X.toarray(), self.y)
            self.estimator.write(getcwd() + '/_tmp/smatrix.dat')

            #self._parameters = self.lasso.coef_
            #self._parameters = cvglmnetCoef(self._fit)[1:, 0]
            self._parameters = self.estimator.parameters

            self._print_training()
            #print(self.estimator.phis.shape)
            #print(f"{len(self.training_data)} alpha: {self.estimator.alpha:.5f} beta: {self.estimator.beta:.3f} gamma: {self.estimator.gamma:.3f}")

        self.iteration += 1

        # Every fit of the forcefield will be slightly different. In case that this is an MPI run
        # broadcast the coefficients to all processes.
        self._parameters = parallel.broadcast(self.parameters)

        # Set the parameters of the model.
        self.model.set_parameters(self.parameters)

        # Write the new parameters to a file.
        np.savetxt(self.settings.parameters_file, self.parameters)
