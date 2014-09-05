"""
Cost functions and their respective derivatives
"""
import numpy

class MSE(object):
    """Mean squared error cost function"""

    def __call__(self, pred, targets):
        n_obs = targets.shape(1); # CHECK DIMS
        err = pred - targets
        return 0.5 * np.sum(err**2) / n_obs

    def grad(self, pred, targets):
        """MSE cost function"""
        return pred - targets

class ExpLL(object):
    """Exponential log likelihood cost function"""

    def __call__(self, pred, targets):
        n_obs = targets.shape(1) # CHECK DIMS
        return np.sum((pred - targets*np.log(pred)))/n_obs;

    def grad(self, pred, targets):
        """Gradient of ExpLL cost function
        """
        return (pred - targets) / (pred * (1 - pred));

class Xent(object):
    """Cross entropy cost function
    """

    def __call__(self, pred, targets):
        nObs = targets.shape(1) # CHECK DIMS
        return -np.sum(targets * np.log(pred) + (1-targets)*np.log(pred))/nObs

    def grad(self, pred, targets):
        """Gradient of cross entropy cost function
        """
        return (pred - targets) / (pred * (1-pred))

class BinClassErr(object):
    """Binary classification error (no gradient)"""

    def __init__(self, thresh=0.5):
        """Initialize classification error with provided threshold
        """
        self.thresh = thresh

    def __call__(self, pred, targets):
        nObs = targets.shape(1) # CHECK DIMS
        return 1 - np.sum(targets == (pred > self.thresh)) / nObs;

    def grad(self, pred=None, targets=None):
        print "No gradient for binary classification error"
        return None

class CorrCoef(object):
    """Correlation coefficient cost function (no gradient)
    """
    def __call__(self, pred, targets):
        return np.corrcoef(pred, targets)

    def grad(self, pred=None, targets=None):
        print 'No gradient for correlation cost'
        return None
        