import functools
import itertools
import numpy as np
import sys

from nose.plugins.skip import SkipTest
from numpy.testing import assert_array_equal


def assert_permutation_equal(p, arr):
    assert_array_equal(list(p), arr)


def skip(func):
    if "--no-skip" in sys.argv:
        return func

    @functools.wraps(func)
    def inner(*args, **kwargs):
        raise SkipTest()
        return func(*args, **kwargs)

    return inner
