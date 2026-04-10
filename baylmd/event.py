from typing import List, TypeVar, Callable
TCallable = TypeVar("TCallable", bound=Callable)

class Event(object):
    def __init__(self) -> None:
        self.listeners: List[TCallable] = list()

    def register(self, listener: TCallable) -> None:
        """ Add an observer to this event. """
        self.listeners.append(listener)

    def __call__(self, *args) -> None:
        """ Call all listeners if there are any. """
        for listener in self.listeners:
            listener(*args)

