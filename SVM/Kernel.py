import numpy.linalg as la
import numpy as np


class Kernel(object):
    """Implements a list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        lamb = 1. / (2 * sigma**2)
        def f(x, y):
            return np.exp(-lamb * la.norm(x - y) ** 2)
        return f 

    @staticmethod
    def _polykernel(dimension, offset):
        def f(x, y):
            return (np.inner(x, y) + offset) ** dimension
        return f

    @staticmethod
    def inhomogenous_polynomial(dimension):
        def f(x, y):
            return (np.inner(x, y) + 1) ** dimension
        return f

    @staticmethod
    def homogenous_polynomial(dimension):
        def f(x, y):
            return np.inner(x, y) ** dimension
        return f

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(np.inner(kappa * x, y) + c)
        return f
