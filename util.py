from scipy.optimize import fmin
import numpy as np


class MonteCarloVariable:

    def __init__(self, mean, std, wildcard=False):
        self._mean = mean
        self._std = std
        self._value = np.random.normal(mean, std)
        self.wildcard = wildcard

    def _wildcard(function):
        """Decorator for MonteCarloVariable
        Re-generates random number before an operation
        """
        def wrapper(*args, **kwargs):
            if args[0].wildcard:
                args[0].generate()
            return function(*args, **kwargs)
        return wrapper

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @_wildcard
    def __float__(self):
        return self.value

    def __repr__(self):
        if self.wildcard:
            return "np.random.normal({}, {})".format(self._mean, self._std)
        else:
            return str(self.value)

    def generate(self):
        self.value = np.random.normal(self._mean, self._std)

    def __add__(self, x):
        return float(self.value) + float(x)

    def __sub__(self, x):
        return float(self.value) - float(x)

    def __mul__(self, x):
        return float(self.value) * float(x)

    def __div__(self, x):
        return float(self.value) / float(x)


def get_intersection(f, g, near_x):
    """Get the intersection of two interpolated functions

    Args
    ----
    f : function 1
    g : function 2
    near_x : a starting point for the solver
    """
    h = lambda x: (f(x) - g(x))**2
    x = fmin(h, near_x, maxfun=1000, disp=False)
    return x, f(x)



