"""Class to store partial expressions for exploration heuristics."""
import numpy as np
from typing import Dict

from dso.program import from_tokens

class Explorer:
    """Class to store partial expressions for exploration heuristics.

    The Cache is a static nested Dict[tuple, Dict[int, int]] to map an expression to expressions with an
    extra symbol along with their frequency.
    i.e. [add, mul, x1] will store as {(add, mul): (x1, 1)}. Of course these will be mapped to the library.
    """
    _cache: Dict[tuple, Dict[int, int]] = {}

    @classmethod
    def get_cache(cls):
        """Return the global cache."""
        return cls._cache

    @classmethod
    def get(cls, item):
        """Get the program by index."""
        return cls._cache[item]

    @classmethod
    def increment(cls, key):
        """Increment the program by index."""
        prefix = key[:-1]
        suffix = key[-1]
        if prefix not in cls._cache:
            cls._cache[prefix] = {}
        cls._cache[prefix][suffix] = cls._cache[prefix].get(suffix, 0) + 1

    @classmethod
    def get_actions_from_program(cls, program: tuple):
        """Get the actions to penalise for a program."""
        if program in cls._cache:
            return cls._cache[program]
        return None

    @classmethod
    def len(cls):
        """Get the number of programs."""
        return len(cls._cache)
