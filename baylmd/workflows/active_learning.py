from typing import List, Optional, Type

from ase import Atoms

from baylmd import parallel

from baylmd.context import Context
from baylmd.forcefields import ForcefieldModel
from baylmd.parsing import parse_bayesian_error
from baylmd.windows import RollingWindow
from baylmd.settings import Settings, DynamicsEnum, ActiveLearningSettings
from baylmd.workflow import Workflow

from baylmd.workflows.molecular_dynamics import MolecularDynamics

from abc import ABC, abstractmethod


class Predicate(ABC):
    """ Abstract base class for a predicate about a Context. """
    def __init__(self, return_value: bool = True) -> None:
        self._return_value: bool = return_value

    @property
    def return_value(self) -> bool:
        """ Immediate return value if the predicate is True. """
        return self._return_value 

    @abstractmethod
    def check(self, context: Context) -> bool:
        """ Check whether the condition is met. """


class RefitInterval(Predicate):
    """ Check whether enough iterations have passed since the last refit. """
    def __init__(self, settings: ActiveLearningSettings) -> None:
        super().__init__(False)
        self._number: int = settings.bayesian_iterations
        
    def check(self, context: Context) -> bool:
        number: int = context.iteration - context.last_refit
        return number < self._number


class MaxThreshold(Predicate):
    """ Check whether the Bayesian error is larger than the maximum allowed threshold. """
    def __init__(self, settings: ActiveLearningSettings) -> None:
        super().__init__()
        self._threshhold: float = settings.bayesian_threshhold

    def check(self, context: Context) -> bool:
        return context.bayesian_error > self._threshhold


class LargerThanAverage(Predicate):
    """ Check whether Bayesian error is N times larger than the average. """
    def __init__(self, settings: ActiveLearningSettings) -> None:
        super().__init__()
        self._factor: float = settings.bayesian_factor

    def check(self, context: Context) -> bool:
        return context.bayesian_error > self._factor * context.mean_error


class VASPThreshold(Predicate):
    """ Check whether Bayesian error is N times larger than the average. """
    def __init__(self, settings: ActiveLearningSettings) -> None:
        super().__init__()
        self._factor: float = 1

    def check(self, context: Context) -> bool:
        return context.bayesian_error > self._factor * context.vasp_criterion


class SkipIterations(Predicate):
    """ Skip the first N iterations. """
    def __init__(self, settings: ActiveLearningSettings) -> None:
        super().__init__(return_value=False)
        self._skip_iterations: int = settings.skip_iterations

    def check(self, context: Context) -> bool:
        return context.iteration < self._skip_iterations


class ActiveLearning(Workflow):
    """ A workflow performing active learning of a forcefield. """
    def __init__(self, structure: Atoms, settings: Settings, forcefield_class: Type = ForcefieldModel, parent: Optional[Workflow] = None) -> None:
        super().__init__(parent=parent)

        self.settings: ActiveLearningSettings = settings.active_learning

        # store some settings necessary for deciding about a refit.
        self.dynamics_mode: DynamicsEnum = settings.dynamics.dynamics_mode

        model = forcefield_class(settings)

        self.molecular_dynamics = MolecularDynamics(structure, settings, model=model, parent=self)

        # Create the running average for the Bayesian error.
        self.average = RollingWindow(settings.active_learning.max_size)
        self.vasp_average = RollingWindow(10)

        # Setup the conditions that determine whether a refit is required.
        self.predicates: List[Predicate] = list()
        for predicate_class in [SkipIterations, RefitInterval, MaxThreshold, LargerThanAverage]:
            self.predicates.append(predicate_class(settings.active_learning))

    def start(self, context: Context) -> None:
        context.iteration = self.molecular_dynamics.iteration
        context.last_refit = self.molecular_dynamics.iteration

        # Read the bayesian error from th log_file.
        if context.iteration > 0:
            errors = parse_bayesian_error(self.molecular_dynamics.log_file)
            for value in errors:
                self.average.update(value)

        #errors = np.loadtxt("dummy.dat")
        #for value in errors[:, 1]:
        #    self.vasp_average.update(value)
        #context.vasp_criterion = self.vasp_average.mean

    def update(self, context: Context) -> None:
        # Perform one step of MD.
        self.molecular_dynamics.update(context)

        sigma = self.molecular_dynamics.sigma()

        context.bayesian_error = sigma.max()

        if context.iteration - context.last_refit == 1:
            self.vasp_average.update(context.bayesian_error)
            context.vasp_criterion = self.vasp_average.mean
            
        # Update the running average.
        self.average.update(context.bayesian_error)
        context.mean_error = self.average.mean

        # Check whether the forcefield should be updated.
        context.refit_required = self.refit_required(context)

        if context.refit_required and parallel.group_comm.rank == 0:
            # Write the current structure to the _tmp directory.
            context.structure.write(f"./_tmp/POSCAR_{parallel.index}")

    @classmethod
    def print_settings(cls, settings: Settings) -> None:
        print(f"\n{cls.__name__}\n")
        if settings.dynamics.dynamics_mode is DynamicsEnum.NEW:
            print("    Number of initial structures:            ", settings.active_learning.initial_iterations)
        print("")
        print("    Iterations for average Bayesian error:   ", settings.active_learning.max_size)
        print("    Iterations between refits:               ", settings.active_learning.bayesian_iterations)
        print("    Emergency refit threshold:               ", settings.active_learning.bayesian_threshhold)
        print("    Prefactor for avg. Bayesian error:       ", settings.active_learning.bayesian_factor)
        print("")

    def refit_required(self, context: Context) -> bool:
        """ Check whether the forcefield should be refitted. """
        for predicate in self.predicates:
            if predicate.check(context):
                return predicate.return_value
        return False
