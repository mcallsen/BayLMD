from typing import TypeVar, Callable, cast

from functools import wraps
from time import time

import ase.parallel as parallel

TCallable = TypeVar("TCallable", bound=Callable)

def timing(function: TCallable) -> TCallable:
    """ A decorator for timing a function. """
    @wraps(function)
    def wrap(*args, **kwargs):
        start_time: float = time()
        result = function(*args, **kwargs)
        end_time: float = time()
        parallel.parprint(f"{function.__name__} took {end_time - start_time: .4f} s.")
        return result
    return cast(TCallable, wrap)
