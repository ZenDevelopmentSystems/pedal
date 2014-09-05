"""Create batches of data 
"""

import numpy as np

class Batches(object):
    """Data batching iterator"""
    def __init__(self, data, targets=None, batch_sz=100):
        n_obs, n_dim = data.shape
        self.n_obs = n_obs
        self.batch_sz = batch_sz
        self.n_batches = np.ceil(n_obs / np.float(batch_sz)).astype(int)
        self.cnt = 0
        self.data = data # COPY?        
        self.targets = targets

    def __call__(self):        
        """Return current batch"""        
        # print 'batch %d / %d' % (self.cnt+1, self.n_batches)

        out_data = self.data[(self.cnt)*self.batch_sz:(self.cnt+1)*self.batch_sz,:]
        self.cnt  = (self.cnt + 1) % self.n_batches 
        if self.targets is not None:
            out_targets = self.targets[(self.cnt)*self.batch_sz:(self.cnt+1)*self.batch_sz]
            return out_data, out_targets
        else:
            return out_data

