"""Preprocessing objects
"""
import numpy as np

class Standardizer(object):
    """Standardizer object: subtract mean and scale by
    standard deviation."""

    def __init__(self):
        self.data_mean = None
        self.data_std = None
        self.n_obs = None
        self.n_dim = None

    def transform(self, data):
        """Standardize data along columns"""
        datamean = data.mean(0)
        self.data_mean = datamean
        datastd = data.std(0)
        self.data_std = datastd
        self.n_obs, self.n_dim = data.shape

        return (data - datamean) / datastd

    def inv_transform(self, data):
        """Inverse transform the standarization transform"""
        return (data * self.data_std) + self.data_mean


class Image(object):

    def __init__(self):
        self.n_obs = None
        self.n_dim = None
        self.dtype0 = None

    def transform(self, data):
        self.n_obs, self.n_dim = data.shape
        self.dtype0 = data.dtype
        return (data/255.).astype('float')

    def inv_transform(self, data):
        return (data * 255.).astype(self.dtype0.name)


class CenterScale(object):

    def __init__(self):
        self.minmax = None
        self.data_min = None
        self.data_max = None
        self.data_mean = None
        self.n_obs = None
        self.n_dim = None

    def transform(self, data, minmax=[-1., 1.]):
        """Center data and scale values to lie within <minmax>
        (TODO)
        """
        pass

    def inv_transform(self, data):
        pass

        


