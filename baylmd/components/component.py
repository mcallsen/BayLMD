from __future__ import annotations

from baylmd.context import Context
from baylmd.settings import Settings

from typing import Protocol, ClassVar, runtime_checkable
from abc import ABC


class IComponentManager(Protocol):
    """ Interface providing functions for adding and retrieving Components. """
    def add_component(self, component_class: type[Component], settings: Settings) -> None:
        """ Add a Component of Type ComponentType. """ 
        ...

    def get_component(self, component_class: type[Component]) -> Component | None:
        """ Get a Component of Type ComponentType. """
        ...

    def get_components(self) -> list[Component]:
        """ Get a list of all components """
        ...


@runtime_checkable
class IContextLogger(Protocol):
    """ Interface for Components that log information. """

    # NOTE: Depending on what components a workflow has, the content of 'ml_std.out' will be
    # different e.g., it might include the Bayesian error or different temperatures. The idea
    # is to assemble the comment line at the beginning of 'ml_std.out' based on all the 
    # COMMENT_STRINGS of the _components, and the line that is written at every MD step using
    # the 'context_to_string' function. 

    COMMENT_STRING: ClassVar[str]

    def context_to_string(self, context: Context) -> str:
        """ Return a string with the information from the context that should be logged. """ 
        ...


class Component(ABC):
    """ Abstract base class for Components that work on a Context. """

    # NOTE: The previous workflows have too many responsibilities and in particular for 
    # MolecularDynamics and ActiveLearning do very similar things. The idea is to break the
    # previous workflows down into smaller components, that can be combined in order to 
    # build a workflow. As far as I can tell, these smaller components should be:
    #
    # (0) Workflow, containing a list of Components representing the previous workflows (Already
    #     implemented in 'workflow.py'.)
    # (1) Forcefield (This one is not strictly necessary, because the Dynamics Component will get 
    #     the forces directly and not from the Context. It will however be used by the Bayesian 
    #     error component to get the A matrix and by the dynamics Component to get the force field.)
    # (2) Dynamics, moving the atoms using ASE.
    # (3) BayesianError, compute the Bayesian error and the trajectory average.
    # (4) Temperature, compute the temperatures for a group of atoms. (For the beginning it is 
    #     okay to have only one temperature Component, but eventually these should be split up. One 
    #     for the total temperature and one for left and right side, respectivily.)
    # (5) Require fitting, checking whether a refit is required.
    # (6) Logger, printing the context to "ml_std.out" (partially done in 'logger.py')
    #
    # (7) Ab initio calculation.
    # (8) Linear regression, training the force field.
    #
    # The previous workflows can then be expressed as combinations of these Components e.g.,
    #
    #   MolecularDynamics:  [Forcefield, Dynamics, Temperature, Logger]
    #   ActiveLearning:     [Forcefield, Dynamics, Temperature, BayesianError, RequireFitting]
    #   ForcefieldTraining: [Forcefield, Abinitio, LinearRegression]
    #
    # This should remove the need for having two seperate scripts to run the workflows.
    #
    # NOTE: There is already a corresponding function 'assemble_workflow' in 'workflow.py'.
    #
    # NOTE: Some Components require other Components can get access to other components using 
    #    
    #     self.workflow.get_component(ComponentType)
    #
    # For example the Bayesian error Component needs access to the Forcefield in order to get the 
    # A matrix.
    #
    # TODO: Implement the components from the list above.
    #
    # TODO: The Component class might need additional methods for printing the settings.

    def __init__(self, workflow: IComponentManager = None) -> None:
        self._workflow = workflow

    @property
    def workflow(self) -> IComponentManager:
        """ Get the parent workflow of this component. """
        return self._workflow

    def start(self, context: Context) -> None:
        """ Perform initial setup before the main loop. """
        # NOTE: Examples for using 'start' would be reading the Bayesian error from 'ml_std.out'
        # in the Bayesian error Component. 
        pass

    def end(self, context: Context) -> None:
        """ Perform additional actions after the main loop. """
        # NOTE: Example usage would be writing POSCAR, if refit is required in the Dynamics Component.
        pass

    def update(self, context: Context) -> None:
        """ Update the context. """
        pass
    