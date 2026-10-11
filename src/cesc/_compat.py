"""Numba shim.

Numba is an optional dependency. When it is installed the scan kernels,
integral images, and connected-component labeling run compiled; without
it the same functions run as plain Python over numpy arrays. Numerical
results are identical; only speed differs. The seasonal runs (hundreds
of 540 x 810 timesteps) need numba; the tests and small domains do not.
"""
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                    # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn
        return wrap

    prange = range
