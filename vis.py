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


class RBMLearning(object):
    def __init__(self, cmap='hot'):
        self.fig = plt.figure("RBM Learning")
        
        self.Wax = self.fig.add_subplot(331, title="W",visible=False)
        self.Wplot = self.Wax.imshow(np.random.rand(2,2),
                                     interpolation='none')

        self.dWax = self.fig.add_subplot(332, title="dW", visible=False)
        self.dWplot = self.dWax.imshow(np.random.rand(2,2),
                                     interpolation='none')

        # PROBABLY A BETTER WAY OF DOING THIS, i.e. AxesStack?
        self.axis_names = ['Wax', 'dWax']

        plt.set_cmap(cmap)

    def close(self):
        plt.close(self.fig)

    def set_data(self, data):
        """Given a dict of axis labels and data, update data"""
        for k, v in data.items():
            try:            
                self.__getattribute__(k).set_data(v)
            except:
                pass
        self.refresh()

    def refresh(self):
        self.fig.canvas.draw()

    def visiblity(self, visibility=True):
        for axname in self.axis_names:
            self.__getattribute__(axname).set_visible(visibility)

        self.refresh()