"""
Activation functions and their corresponding derivatives
"""

import numpy as np

tiny = np.finfo(np.single).tiny
cutoff = np.log(tiny)

def stabilize_input(inp, k=1.):
    """Ensure numerical stability"""
    inp[inp * k > -1.*cutoff] = -cutoff/k
    inp[inp * k < cutoff] = cutoff/k
    return inp

class Linear(object):
    """Linear activation function"""

    def __call__(self, z):
        return z

    def grad(self, act):
        """Gradient of linear activation function"""
        return np.ones_like(act)

class LinearSaturated(object):
    """Linear activation function"""

    def __init__(self, k=1, out_min=0, out_max=1):
        self.k = k
        self.min = out_min
        self.max = out_max

    def __call__(self, z):

        zc = z.copy()*self.k
        zc[zc < self.min*self.k] = self.min*self.k
        zc[zc > self.max*self.k] = self.max*self.k
        return zc

    def grad(self, act):
        """Gradient of linear saturated activation function"""
        df = np.zeros_like(act)
        df[(act < self.max*self.k) & (act > self.min*self.k)] = 1
        return df


class Sigmoid(object):
    """Sigmoid activation function"""

    def __call__(self, z):
        return 1 / (1 + np.exp(-1.*z))

    def grad(self, act):
        """Gradient of Sigmoid Activation
        """
        return act*(1.0-act)

class Tanh(object):
    """Hyperbolic tangent activation function
    """
    def __call__(self, z):
        return np.tanh(z)

    def grad(self, act):
        """Gradient of Hyperbolic tangent activation function
        """
        return (1 - act**2)

class Exponential(object):
    """Exponential activation function
    """
    def __call__(self, z):
        return np.exp(stabilize_input(z,1))

    def grad(self, act):
        """Gradient of exponential activation function
        """
        return stabilize_input(act,1)

class Softrect(object):
    """Soft rectification activaiton function"""

    def __init__(self, k=8):
        self.k = k

    def __call__(self, z):
        z = stabilize_input(z, self.k)
        return (1./ self.k) * np.log(1. + np.exp(self.k * z))

    def grad(self, act):
        act = stabilize_input(act, self.k)
        return 1./(1. + np.exp(-self.k * act));

class Softmax(object):
    """Softmax activation function (Multinomial classification)"""

    def __call__(self, z):
        z = z.T
        max_z = z.max(0)
        tmp = np.exp(z - max_z)
        return (tmp / tmp.sum(0)).T

    def grad(self,z):
        print "No gradient for Softmax"
        return None

            