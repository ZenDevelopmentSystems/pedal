"""Params / required parameters for various object classes used in PEDAL
   (Also good cause I can't keep track of them)
"""


class Params(object):
    def __init__(self):
        pass

    def show(self):
        """Display attributes of current parameter object"""
        for k, v in self.__dict__.items():
            print "{0:18s}{1:18s}".format(str(k)+':', str(v))

    def update(self, new_params):
        """Update the state of a Params object based on the
        key-value updates provided byt he dict() <new_params>.
        """
        for k, v in new_params.items():            
            setattr(self,k,v)

class SGDParams(Params):
    def __init__(self):
        self.train_type = 'sgd'
        self.n_epoch = 100
        self.lrate = 0.1
        self.w_penalty = 0.01
        self.begin_wd = 0
        self.momentum = 0.
        self.begin_anneal = 1e10
        self.pcd = False
        self.sparsity = 0.
        self.sparse_gain = 0.               

        self.batch_sz=100

        self.dropout = 0.        
        self.verbose = False
        self.visualize = False
        self.display_every = 1e10
        self.save_every = 1e10
        self.save_fold = None

        self.topo_reg = 0.
        self.reg_strength = 0.5
        self.log = {}

class RBMParams(Params):
    def __init__(self):
        self.model_class = 'rbm'
        self.input_type = 'binary'
        self.n_vis = 100
        self.n_hid = 10
        self.rlu = False
        self.sample_hid = True
        self.sample_vis = True
        self.is_classifier = False
        self.n_gibbs = 1
        self.params = ['W','b','a']

class DBNParams(Params):
    def __init__(self):
        self.model_class = 'dbn'



class SGDNNParams(Params):
    def __init__(self):
        self.n_epoch = 100
        self.lrate = 0.1
        self.w_decay = 0.01
        self.begin_wd = 1
        self.momentum = 0.
        self.begin_anneal = 1e10
        self.pcd = False
        self.sparsity = 0.
        self.sparse_gain = 0.
        self.n_gibbs = 1

        self.batch_sz=100

        self.dropout = 0.        
        self.verbose = False
        self.display_every = 1e10
        self.save_every = 1e10
        self.save_fold = None

        self.topo_reg = 0.

class NNParams(Params):
    def __init__(self):
        self.model_class = 'nn'

