from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from baylmd.context import Context
from baylmd.event import Event

class Workflow(ABC):
    """ Abstract base class for Workflows. """
    def __init__(self, parent: Optional[Workflow] = None) -> None:
        super().__init__()
        self.parent: Optional[Workflow] = parent
        self.on_update: Event = Event()

    def start(self) -> None:
        pass

    def end(self) -> None:
        pass

    @abstractmethod
    def update(self, context: Context) -> None:
        """ Perform a single step of the workflow. """

