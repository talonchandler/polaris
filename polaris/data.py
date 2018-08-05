from polaris import util, viz
import numpy as np
import os
import tifffile

class Data:
    """
    A Data object represents all of the data a multiview polarized light 
    microscope can collect stored in a 5D array of values [x, y, z, pol, view].
    A Data object is a discretized member of data space V. 
    """
    def __init__(self, g=np.zeros((10,10,10,4,2), dtype=np.float32),
                 vox_dim=(100,100,100),
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
        print('Saving '+filename)
        row_labels = 'View = ' + np.apply_along_axis(util.xyz2str, 1, self.views)
        col_labels = 'Polarizer = ' + np.apply_along_axis(util.xyz2str, 2, self.pols)
        self.yscale = 1e-3*self.vox_dim[1]*self.g.shape[1]
        yscale_label = str(self.yscale) + ' $\mu$m'
        viz.plot5d(filename, self.g, row_labels, col_labels, yscale_label, normalize=normalize)

    def save_tiff(self, folder):
        print('Writing '+folder)
        
        for f in [folder, folder+'SPIMA', folder+'SPIMB']:
            if not os.path.exists(f):
                os.makedirs(f)

        for i, view in enumerate(['SPIMA', 'SPIMB']):
            for j in range(4):
                filename = folder + view + '/' + view + '_reg_' + str(j) + '.tif'
                data = self.g[...,j,i]
                data = np.swapaxes(data, 0, 2)
                with tifffile.TiffWriter(filename, imagej=True) as tw:
                    tw.save(data[None,:,None,:,:,None]) # TZCYXS

    def read_tiff(self, folder, roi=None):
        for i, view in enumerate(['SPIMA', 'SPIMB']):
            for j in range(4):
                filename = folder + view + '/' + view + '_reg_' + str(j) + '.tif'
                with tifffile.TiffFile(filename) as tf:
                    print('Reading '+filename)
                    data = tf.asarray() # ZYX
                    if roi is not None: # Crop
                        data = data[roi[2,0]:roi[2,1], roi[1,0]:roi[1,1], roi[0,0]:roi[0,1]]
                    if self.g.shape[0] != data.shape[2]: # Make g
                        datashape = (data.shape[2], data.shape[1], data.shape[0], self.pols.shape[1], self.pols.shape[0])
                        self.g = np.zeros(datashape, dtype=np.float32)
                    if data.dtype == np.uint16: # Convert 
                        data = (data/np.iinfo(np.uint16).max).astype(np.float32)
                    self.g[...,j,i] = np.swapaxes(data, 0, 2)
