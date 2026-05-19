from baylmd.context import Context
from baylmd.settings import Settings
from baylmd.components.component import Component, IComponentManager

from typing import Iterable

class Workflow(Component):
    """ 
    Component implementing the ComponentManager Protocoll.

    This Component holds a list of Components that are all applied to the context, when any of
    'start', 'update', and 'end' are called. A Workflow with a specific functionality can be 
    assembled by adding the required components to implement that functionality.
    
    Can pass another ComponentManager to the constructor, which allows for nested workflows.
    In the current implementation the Components are stored in a Dictionary by their type. This
    might need to be changed once there is a use for having multiple Components of the same type.
    """
    def __init__(self, workflow: IComponentManager = None) -> None:
        super().__init__(workflow=workflow)
        self._components: dict[type[Component], Component] = dict()

    def add_component(self, component_class: type[Component], settings: Settings) -> None:
        self._components[component_class] = component_class(self, settings)
        return self

    def get_component(self, component_class: type[Component]) -> Component | None:
        return self._components.get(component_class)

    def get_components(self) -> list[Component]:
        return self._components.values()

    def start(self, context: Context) -> None:
        # Call the start method of every component.
        for component in self._components.values():
            component.start(context)

    def update(self, context: Context) -> None:
        # Apply all Components to the context.
        for component in self._components.values():
            component.update(context)

    def end(self, context: Context) -> None:
        # Call the end method of every component.
        for component in self._components.values():
            component.end(context)


def assemble_workflow(component_classes: Iterable[type[Component]], settings: Settings) -> Workflow:
    """ Assemble a Workflow from a list of Component classes. """

    # NOTE: This function can be used in 'active_learning.py' or 'scripts/run_active_learning.py'
    # to assemble the corresponding workflow. The list of Component types to add, should be 
    # determined from 'Settings'.

    workflow = Workflow()
    for component_class in component_classes:
        workflow.add_component(component_class, settings)
    return workflow