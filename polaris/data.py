from polaris import util, viz
import numpy as np

class Data:
    """
    A Data object represents all of the data a multiview polarized light 
    microscope can collect stored in a 5D array of values [x, y, z, pol, view].
    A Data object is a discretized member of data space V. 
    """
    def __init__(self, g=np.zeros((10,10,10,4,2)), vox_dim=(100,100,100),
                 pols=np.array([[[0,0,-1], [0,1,-1], [0,1,0], [0,1,1]],
                                [[1,0,0], [1,1,0], [0,1,0], [-1,1,0]]]),
                 views=np.array([[0,0,1],[1,0,0]])):

        self.g = g # NX X NY X NZ X P X V
        self.NX = self.g.shape[0]
        self.NY = self.g.shape[1]
        self.NZ = self.g.shape[2]
        self.P = self.g.shape[3]        
        self.V = self.g.shape[4]
        self.vox_dim = vox_dim

        self.pols = pols
        self.pols_norm = pols/np.linalg.norm(pols, axis=2)[:,:,None] # V X P X 3
        self.views = views # V x 3
        self.yscale = 1e-3*vox_dim[1]*self.g.shape[1]

    def save_mips(self, filename='mips.pdf', normalize=False):
        row_labels = 'View = ' + np.apply_along_axis(util.xyz2str, 1, self.views)
        col_labels = 'Polarizer = ' + np.apply_along_axis(util.xyz2str, 2, self.pols)
        yscale_label = str(self.yscale) + ' $\mu$m'
        viz.plot5d(filename, self.g, row_labels, col_labels, yscale_label, normalize=normalize)
