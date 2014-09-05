"""Restricted Boltzmann machine module"""

# TODO: 
# REGULARIZATION
# MULTINOMIAL
# ADD JOINT CATEGORY MODELING

import numpy as np
from params import SGDParams, RBMParams
from batches import Batches
import time

def draw_normal(mu):
    return np.random.normal(mu)

def draw_bernoulli(p):
    return (p > np.random.rand()).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-1.*z))

def calc_reg_mat(p_hid):
    return 1.

class RBMTrainer(SGDParams):
    """Stochastic Gradient Descent Trainer"""
    def __init__(self, rbm, params):
        super(RBMTrainer, self).__init__()

        if params is None:
            params = SGDParams() 

        for k, v in params.__dict__.items():            
            setattr(self,k,v)

        self.rbm = rbm

        # (FOR MOMENTUM)
        self.dW_old = np.zeros_like(rbm.W)
        self.da_old = np.zeros_like(rbm.a)
        self.db_old = np.zeros_like(rbm.b)

        self.log['error'] = []
        self.log['lrate'] = []

    def annealing(self):
        pass

    def print_progress(self):
        print '\nEpoch %d' % (self.current_epoch + 1)
        print 'Weight penalty: %f' % self.w_penalty
        print 'Learning rate: %f' % self.log['lrate'][-1]
        print 'Recon. error: %f' % self.log['error'][-1]
        out_str = [];
        for param in self.rbm.params:
            print "{0:5s} = {1:5f}".format('|'+ param +'|', 
                                          np.linalg.norm(self.rbm.__getattribute__(param)))

    def vis_learning(self):
        print self.rbm.W.shape
        # plt.figure(1)
        # plt.clf()
        # w = self.rbm.W[:,0]
        # plt.imshow(w.reshape(28,28),cmap='gray',interpolation='none')
        # plt.show()
        # time.sleep(0.05)


    def log_info(self,err):
        self.log['error'].append(err)
        self.log['lrate'].append(self.lrate)

    def train(self, data):
        """Train RBM using stochastic graident descent"""

        w_penalty0 = self.w_penalty
        lrate0 = self.lrate

        # TRANSFORM DATA INTO "ITERABLE" BATCHES
        data = Batches(data, batch_sz=self.batch_sz)
        states = None
        for self.current_epoch in xrange(self.n_epoch):

            # ANNEALING
            if self.current_epoch > self.begin_anneal:
                self.lrate = max(lrate0*((self.current_epoch- 
                                        self.begin_anneal)**-.25),1e-8)


            # WEIGHT DECAY SCHEDULE
            if self.current_epoch < self.begin_wd:
                self.w_penalty = 0
            else:
                self.w_penalty = w_penalty0

            # LOOP OVER BATCHED DATA
            batch_err_tot = 0
            while True:
                states, batch_err = self.run_gibbs(data(), states)
                gradients = self.calc_gradients(**states)
                self.update_params(**gradients)
                batch_err_tot += batch_err

                if data.cnt == data.n_batches - 1:
                    break

            self.log_info(batch_err_tot)

            if (self.verbose):
                self.print_progress()

            if self.visualize & (self.current_epoch % self.display_every == 0):
                self.log['gradients'] = gradients
                self.log['states'] = states
                self.vis_fun.vis(self)

        return self.rbm, self.log
        

    def run_gibbs(self, batch_data, states):
        """Runn Gibbs sampler for RBM (i.e. Contrastive Divergence)"""   

        # CALCULATE DATA-DRIVEN HIDDEN-UNIT REGULARIZATION
        a_hid, reg_mat = self.rbm.hgv(batch_data, 
                                      self.rbm.sample_hid, calc_reg=True)

        if self.pcd and (states != None):            
            a_hid = states['a_hid']
            
        a_vis0 = batch_data.copy()                
            
        # REGULARIZE DATA-DRIVEN HIDDEN UNIT ACTIVATIONS
        a_hid0 = (1. - self.reg_strength) * a_hid + self.reg_strength*reg_mat

        gibbs_cnt = 1
        while True:
            # GO DOWN
            a_vis =  self.rbm.vgh(a_hid, self.rbm.sample_vis)

            # GO BACK UP
            if gibbs_cnt == self.rbm.n_gibbs:
                a_hid, _ = self.rbm.hgv(a_vis,0);
                break
            else:
                a_hid, _ = self.rbm.hgv(a_vis, self.rbm.sample_hid);
            
            gibbs_cnt += 1

        # PACKAGE STATES INTO DICT
        states = {'a_vis0': a_vis0, 'a_hid0': a_hid0, 'a_vis': a_vis, 'a_hid': a_hid}
        recon_error = ((a_vis - batch_data)**2).sum()       

        return states, recon_error

    def calc_gradients(self, a_vis0, a_hid0, a_vis, a_hid):
        """Calculate Boltzmann machine gradients with repsect to model params"""       
        
        dW = (np.dot(a_vis0.T, a_hid0) - np.dot(a_vis.T, a_hid))/a_vis0.shape[0]      
        da = a_hid.mean(0)
        db = a_vis.mean(0)

        # PACKAGE GRADIENTS 
        gradients = {'dW': dW, 'da': da, 'db': db}

        return gradients

    def update_params(self, dW, da, db):
        dW = self.momentum*self.dW_old + (1-self.momentum)*dW
        self.rbm.W += self.lrate*dW        

        # WEIGHT REGULARIZATION
        if self.w_penalty > 0: # (L2)
            W_penalty = -self.lrate*self.w_penalty*self.rbm.W
        elif self.w_penalty < 0: # (L1)
            W_penalty = self.lrate*self.w_penalty*np.sign(self.rbm.W)
        else:
            W_penalty = 0
        self.rbm.W += W_penalty

        db = self.momentum*self.db_old + (1-self.momentum)*db
        self.rbm.b += db

        da = self.momentum*self.da_old + (1-self.momentum)*da
        self.rbm.a += da

        # (MOMENTUM MEMORY)
        self.dW_old = dW
        self.da_old = da
        self.db_old = db

class RBM(RBMParams):
    """General Restricted Boltzmann Machine object"""
    def __init__(self, params=None):
        super(RBM, self).__init__()       

        if params is None:
            params = RBMParams() # DEFAULT PARAMETERS OBJECT

        for k, v in params.__dict__.items():            
            setattr(self,k,v)

        w_scale =  np.sqrt(6. / (2 * self.n_vis));
        self.W = w_scale*(np.random.rand(self.n_vis, self.n_hid) - 0.5)
        self.a = np.zeros(self.n_hid)
        self.b = np.zeros(self.n_vis)

        if self.input_type == 'binary':
            self.sample_fun = draw_bernoulli
        else:
            self.sample_fun = draw_normal

    def hgv(self, vis_state, sample_hid=False, calc_reg=False):
        """Return binary hidden unit states given visible states
        """
        # p(h | v)
        a_hid = sigmoid(np.dot(vis_state, self.W) + self.a)

        # REGULARIZE HIDDEN UNIT STATES
        if calc_reg:
            reg_mat = calc_reg_mat(a_hid)
        else:
            reg_mat = None
        
        # SAMPLE OR MEAN FIELD?
        if sample_hid:
            a_hid = draw_bernoulli(a_hid)

        return a_hid, reg_mat

    def vgh(self, hid_states, sample_vis=False):
        """Return visible unit states given hidden
        """
        if self.input_type == 'binary':
            a_vis = sigmoid(np.dot(self.W, hid_states.T) + self.b[:,None])

        elif self.input_tyype == 'gaussian':
            a_vis = np.dot(self.W, hid_states.T) + self.b[:,None]

        if sample_vis:
            a_vis = self.sample_fun(a_vis)

        return a_vis.T

    def draw_samples(self, a_vis0, n_samples=50, h_mask=1.):
        """Draw <n_samples> from current model given initial visible
        states <a_vis0>. Can also anchor specific hidden units with
        <h_mask>"""
        pass




class Conditional(RBM):
    """Conditional RBM (aka Dynamic RBM) under construction"""
    def __init__(self, params):        
        super(Conditional, self).__init__()

        for k, v in params.iteritems():
            if hasattr(self, k):
                setattr(self,k,v)

        if self.input_type == 'binary':
            self.sample_fun = draw_bernoulli
        else:
            self.sample_fun = draw_normal

        self.params = ['W','b','a','d']


class MCRBM(RBM):
    """Mean Covariance RBM under construction"""
    def __init__(self, params):
        super(MCRBM, self).__init__()

        for k, v in params.iteritems():
            if hasattr(self, k):
                setattr(self,k,v)


if __name__ == '__main__':
    r = RBM(n_vis,n_hid)
