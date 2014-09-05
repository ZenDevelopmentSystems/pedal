"""Visualization objects
"""
import numpy as np
import matplotlib 
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt



def gallery_gray_patches(W,show=False, rescale=False):
    """Create a gallery of image patches from <W>, with
    grayscale patches aligned along columns"""

    n_vis, n_feats = W.shape;

    n_pix = np.sqrt(n_vis)
    n_rows = np.floor(np.sqrt(n_feats))
    n_cols = np.ceil(n_feats / n_rows)    
    border_pix = 1;

    # INITIALIZE GALLERY CONTAINER    
    im_gallery = np.nan*np.ones((border_pix+n_rows*(border_pix+n_pix),
                   border_pix+n_cols*(border_pix+n_pix)))

    for iw in xrange(n_feats):
        # RESCALE EACH IMAGE
        W_tmp = W[:,iw].copy()
        if rescale:
            W_tmp = (W_tmp - W_tmp.mean())/np.max(np.abs(W_tmp)); 

        W_tmp = W_tmp.reshape(n_pix, n_pix)

        # FANCY INDEXING INTO IMAGE GALLERY
        im_gallery[border_pix + np.floor(iw/n_cols)*(border_pix+n_pix): 
                        border_pix + (1 + np.floor(iw/n_cols))*(border_pix+n_pix) - border_pix, 
                  border_pix + np.mod(iw,n_cols)*(border_pix+n_pix): 
                        border_pix + (1 + np.mod(iw,n_cols))*(border_pix+n_pix) - border_pix] = W_tmp
    if show:
        plt.imshow(im_gallery,interpolation='none')
        plt.axis("image")
        plt.axis("off")

    return im_gallery



def gallery_rgb_patches(W,show=False, rescale=False):
    """Create a gallery of image patches from <W>, with
    rgb patches aligned along columns"""
    
    n_vis, n_feats = W.shape;

    n_pix = np.sqrt(n_vis/3)
    n_rows = np.floor(np.sqrt(n_feats))
    n_cols = np.ceil(n_feats / n_rows)    
    border_pix = 1;

    # INITIALIZE GALLERY CONTAINER    
    im_gallery = np.nan*np.ones((border_pix+n_rows*(border_pix+n_pix),
                   border_pix+n_cols*(border_pix+n_pix),3))

    for iw in xrange(n_feats):
        # RESCALE EACH IMAGE
        W_tmp = W[:,iw].copy()        
        W_tmp = W_tmp.reshape(3, n_pix, n_pix).transpose([1,2,0])

        if rescale:
            for c in xrange(3):
                cols = W_tmp[:,:,c]
                W_tmp[:,:,c] = (cols - cols.mean())/np.max(np.abs(cols))


        # FANCY INDEXING INTO IMAGE GALLERY
        im_gallery[border_pix + np.floor(iw/n_cols)*(border_pix+n_pix): 
                        border_pix + (1 + np.floor(iw/n_cols))*(border_pix+n_pix) - border_pix, 
                  border_pix + np.mod(iw,n_cols)*(border_pix+n_pix): 
                        border_pix + (1 + np.mod(iw,n_cols))*(border_pix+n_pix) - border_pix,:] = W_tmp
    if show:
        plt.imshow(im_gallery,interpolation='none')
        plt.axis("image")
        plt.axis("off")


class RBMTraining(object):
    """General RBM Training Object"""
    def __init__(self, cmap='jet'):

        self.fig = plt.figure("RBM Learning")
        
        # INITIALIZE PLOTS
        self.W_ax = self.fig.add_subplot(221, title="W")
        self.W_plot = self.W_ax.imshow(np.random.rand(2,2),
                                     interpolation='none')

        self.dW_ax = self.fig.add_subplot(222, title="dW")
        self.dW_plot = self.dW_ax.imshow(np.random.rand(2,2),
                                     interpolation='none')

        self.a_hid_ax = self.fig.add_subplot(223, title="Hidden Activations")
        self.a_hid_plot = self.a_hid_ax.hist(np.random.rand(200),bins=25)

        self.err_ax = self.fig.add_subplot(224, title="Recon. Error")
        self.err_plot = self.err_ax.plot(range(10),'r')


        # PROBABLY A BETTER WAY OF DOING THIS, i.e. AxesStack?
        self.axis_names = ['W_ax', 'dW_ax','a_hid_ax', 'err_ax']

    def close(self):
        plt.close(self.fig)

    def set_data(self, data):
        """Given a dict of axis labels and data, update axes"""
        for k, v in data.items():
            print ('key', k, 'value', v)
            try:            
                self.__getattribute__(k).set_data(v)
            except: 
                self.__getattribute__(k).set_ydata(v)
            else:
                print 'data update failed: %s' % k

        self.refresh()

    def refresh(self):
        self.fig.canvas.draw()
        # self.fig.show()

    def visibility(self, visibility=True):
        print self.axis_names
        for ax_name in self.axis_names:
            try:
                self.__getattribute__(ax_name).set_visible(visibility)
            except: pass

        self.refresh()

class RBMTrainingMNIST(RBMTraining):
    def __init__(self):
        super(RBMTrainingMNIST, self).__init__()
    plt.set_cmap("gray")
    def vis(self, trainer):
        print 'updating data'
        data = {'W_plot': gallery_gray_patches(trainer.rbm.W), 
                'dW_plot': gallery_gray_patches(trainer.log['gradients']['dW'])}
                # 'err_plot': np.array(trainer.log['error'])}
                # 'a_hid_plot': trainer.log['states']['a_hid']}

        print 'setting data/ refreshing'
        self.set_data(data)
        self.refresh()

