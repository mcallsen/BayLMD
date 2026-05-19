from baylmd.components.component import Component, IComponentManager, IContextLogger
from baylmd.context import Context

from typing import Iterable

class Logger(Component):
    def __init__(self, workflow: IComponentManager = None) -> None:
        super().__init__(workflow=workflow)

        # Get a list of all the components that log information from the workflow.
        self._components: list[IContextLogger] = filter_context_loggers(self.workflow.get_components())
        
        # TODO: Set self._file_name, which will be used in update later. The initialisation of the
        # log file should happen somewhere else, because more then one Component relies on it.

    def update(self, context: Context):
        # TODO: Compose the line using all the _components.context_to_string(context) functions.
        # and write it to the log file.
        pass


def filter_context_loggers(components: Iterable[Component]) -> list[IContextLogger]:
    """ Filter all ContextLoggers from a list of Components. """
    return [component for component in components if isinstance(component, IContextLogger)]