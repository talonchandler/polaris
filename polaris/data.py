from polaris import util, viz
import numpy as np
import os
import tifffile
import logging
log = logging.getLogger('log')

class Data:
    """
    A Data object represents all of the data a multiview polarized light 
    microscope can collect stored in a 5D array of values [x, y, z, pol, view].
    A Data object is a discretized member of data space V. 
    """
    def __init__(self, g=np.zeros((10,10,10,4,2), dtype=np.float32),
                 vox_dim=[130,130,130],
                 ill_nas=2*[0], det_nas=2*[0.8],
                 ill_optical_axes=[[1,0,0], [0,0,1]],
                 det_optical_axes=[[0,0,1], [1,0,0]],
                 pols=np.array([[[0,0,-1], [0,1,-1], [0,1,0], [0,1,1]],
                                [[1,0,0], [1,1,0], [0,1,0], [-1,1,0]]])):

        self.g = g
        self.X = g.shape[0]
        self.Y = g.shape[1]
        self.Z = g.shape[2]
        self.P = g.shape[3]        
        self.V = g.shape[4]
        self.vox_dim = vox_dim
        self.pols = pols
        self.pols_norm = pols/np.linalg.norm(pols, axis=2)[:,:,None] # V X P X 3
        self.ill_nas = ill_nas
        self.det_nas = det_nas
        self.ill_optical_axes = ill_optical_axes
        self.det_optical_axes = det_optical_axes

    def remove_background(self, level=None):
        log.info('Applying background correction')
        # By default subtract the average of a 10x10x10 ROI with corner at
        # (5,5,5) --- not too close to the corner
        if level is None:
            B0 = np.mean(self.g[5:15,5:15,5:15,:,0])
            B1 = np.mean(self.g[5:15,5:15,5:15,:,1])
            log.info('A background: '+str(B0))
            log.info('B background: '+str(B0))
            self.g[...,0] = self.g[...,0] - B0 
            self.g[...,1] = self.g[...,1] - B1
        else:
            self.g[...,0] = self.g[...,0] - level[0]
            self.g[...,1] = self.g[...,1] - level[1]

    def apply_calibration_correction(self, epi_cal, ls_cal, order, lake_response):
        log.info('Applying calibration correction')
        epi_normed = epi_cal/np.mean(epi_cal, axis=-1)[:,None]
        ls_corrected = ls_cal/epi_normed
        cal_data = ls_corrected/np.mean(ls_corrected, axis=-1)[:,None]
        cal_data[0,:] = cal_data[0,order[0][:]] # reorder
        cal_data[1,:] = cal_data[1,order[1][:]] # reorder
        for p in range(self.P):
            for v in range(self.V):
                correction = lake_response[v,p]/cal_data[v,p]
                log.info('Calibration V'+str(v)+'P'+str(p)+' correction '+str(correction))
                self.g[...,p,v] = self.g[...,p,v]*correction

    def apply_power_correction(self, correction=None):
        if correction is None:
            # correction = (self.det_nas[0]/self.det_nas[1])**2 # Naive
            haA = np.arcsin(self.det_nas[0]/1.33)
            haB = np.arcsin(self.det_nas[1]/1.33)
            correction = (1 - np.cos(haA))/(1 - np.cos(haB))
        log.info('Power correction '+str(correction))
        self.g[...,0] = self.g[...,0]/correction

    def apply_voxel_size_correction(self, vox_dim_A, vox_dim_B):
        correction = np.prod(np.array(vox_dim_B)/np.array(vox_dim_A))
        log.info('Voxel size correction '+str(correction))
        self.g[...,0] = correction*self.g[...,0]

    def apply_padding(self, width=2):
        log.info('Padding data with '+str(width)+' zeros')
        pads = [(width,width), (width,width), (width,width), (0,0), (0,0)]
        self.g = np.pad(self.g, pads, mode='edge')

    def apply_depadding(self, width=2):
        log.info('Depadding after 3D inverse Fourier transform')             
        self.g = self.g[width:-width, width:-width, width:-width, :, :]
            
    def save_mips(self, filename='mips.pdf', normalize=False):
        log.info('Saving '+filename)
        if np.min(self.g) < -1e-3:
            log.info('Warning: minimum data is ' + str(np.min(self.g)) + '. Truncating negative values in '+filename)

        row_labels = 'Illumination axis = ' + np.apply_along_axis(util.xyz2str, 1, self.ill_optical_axes) + '\n Detection axis = ' + np.apply_along_axis(util.xyz2str, 1, self.det_optical_axes) + '\n NA = ' + util.f2str(self.det_nas)
        col_labels = 'Polarizer = ' + np.apply_along_axis(util.xyz2str, 2, self.pols)
        self.yscale = 1e-3*self.vox_dim[1]*self.g.shape[0]
        yscale_label = '{:.2f}'.format(self.yscale) + ' $\mu$m'
        viz.plot5d(filename, self.g.clip(min=0), row_labels, col_labels, yscale_label, normalize=normalize)

    def save_tiff(self, folder, diSPIM_format=True):
        log.info('Writing '+folder)

        if diSPIM_format:
            # Same format as shroff lab
            for f in [folder, folder+'SPIMA', folder+'SPIMB']:
                if not os.path.exists(f):
                    os.makedirs(f)

            for i, view in enumerate(['SPIMA', 'SPIMB']):
                for j in range(4):
                    filename = folder + view + '/' + view + '_reg_' + str(j) + '.tif'
                    data = self.g[...,j,i].astype(np.float32)
                    data = np.swapaxes(data, 0, 2)
                    with tifffile.TiffWriter(filename, imagej=True) as tw:
                        tw.save(data[None,:,None,:,:,None]) # TZCYXS
        else:
            filename = folder + 'data.tif'
            data = np.moveaxis(self.g, [4, 2, 3], [0, 1, 2])
            with tifffile.TiffWriter(filename, imagej=True) as tw:
                tw.save(data[:,:,:,:,:,None]) # TZCYXS

    def read_tiff(self, folder, roi=None, order=None):
        log.info('ROI: ' + str(roi))
        for i, view in enumerate(['SPIMA', 'SPIMB']):
            for j in range(4):
                filename = folder + view + '/' + view + '_reg_' + str(j) + '.tif'
                with tifffile.TiffFile(filename) as tf:
                    log.info('Reading '+filename)
                    data = tf.asarray() # ZYX order
                    if roi is not None: # Crop
                        data = data[roi[2][0]:roi[2][1], roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]]
                    if self.g.shape[0] != data.shape[2]: # Make g with correct shape
                        datashape = (data.shape[2], data.shape[1], data.shape[0], self.pols.shape[1], self.pols.shape[0])
                        self.g = np.zeros(datashape, dtype=np.float32)
                    if data.dtype == np.uint16: # Convert 
                        data = (data/np.iinfo(np.uint16).max).astype(np.float32)
                    if order is not None:
                        jj = order[i][j]
                    else:
                        jj = j
                    self.g[...,jj,i] = np.swapaxes(data, 0, 2) # XYZPV order
