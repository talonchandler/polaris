import matplotlib.pyplot as plt
from polaris import util, viz, spang
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
        self.P = pols.shape[1]
        self.V = pols.shape[0]
        self.vox_dim = vox_dim
        self.pols = pols
        self.pols_norm = pols/np.linalg.norm(pols, axis=2)[:,:,None] # V X P X 3
        self.ill_nas = ill_nas
        self.det_nas = det_nas
        self.ill_optical_axes = ill_optical_axes
        self.det_optical_axes = det_optical_axes

    def remove_background(self, percentile=None):
        log.info('Applying background correction')
        # By default subtract the average of a 10x10x10 ROI with corner at
        # (5,5,5) --- not too close to the corner
        if percentile is None:
            B0 = np.mean(self.g[5:15,5:15,5:15,:,0])
            B1 = np.mean(self.g[5:15,5:15,5:15,:,1])
            log.info('A background: '+str(B0))
            log.info('B background: '+str(B0))
            self.g[...,0] = self.g[...,0] - B0 
            self.g[...,1] = self.g[...,1] - B1
        else:
            xperc = np.percentile(self.g, percentile)
            self.g -= xperc
            log.info('Removing '+str(percentile)+'th percentile bkg = '+str(np.round(xperc,2)))

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
                for j in range(self.g.shape[3]):
                    filename = folder + view + '/' + view + '_reg_' + str(j) + '.tif'
                    data = self.g[...,j,i].astype(np.float32)
                    data = np.swapaxes(data, 0, 2)
                    with tifffile.TiffWriter(filename, imagej=True) as tw:
                        tw.save(data[None,:,None,:,:,None]) # TZCYXS
        else:
            if not os.path.exists(folder):
                    os.makedirs(folder)
            filename = folder + 'data.tif'
            data = np.moveaxis(self.g, [4, 2, 3], [0, 1, 2])
            with tifffile.TiffWriter(filename, imagej=True) as tw:
                tw.save(data[:,:,:,:,:,None]) # TZCYXS

    def visualize(self, filename, **kwargs):
        # Calculate arrow positions
        X = np.float(self.g.shape[0])
        Y = np.float(self.g.shape[1])
        Z = np.float(self.g.shape[2])# - kwargs['z_shift']
        max_dim = np.max([X,Y,Z])

        tips = np.array([[(X-max_dim)/2,Y/2,Z/2],
                         [X/2,Y/2,(Z+max_dim)/2]])
        rr = 10
        
        # Visualize each volume
        for i in range(self.g.shape[4]):
            for j in range(self.g.shape[3]):
                o = np.array([X/2, Y/2, Z/2])
                pol = self.pols[i,j,:]
                rexc = tips[i,:] - o
                rdet = tips[(i+1)%2,:] - o
                tip = np.cross(rdet, pol)
                tip_n = tip*np.linalg.norm(rexc)/np.linalg.norm(tip)
                if np.dot(tip_n, rexc) < 0:
                    tip_n *= -1
                
                arrows = np.array([[tip_n+o,rr*pol],
                                   [tip_n+o,-rr*pol]])
                
                sp = spang.Spang(f=np.zeros(self.g.shape[0:3] + (15,)), vox_dim=self.vox_dim) # For visuals only
                sp.f[...,0] = self.g[...,j,i]
                sp.visualize(filename+'v'+str(i)+'p'+str(j), viz_type='Density',
                             arrows=arrows, arrow_color=np.array([1,0,0]),
                             **kwargs)
    
    def read_tiff(self, folder, roi=None, order=None, format='diSPIM',
                  tilts=['0', '1', '-1'], filenum='0', bkg_perc=10):
        log.info('ROI: ' + str(roi))
        if format == 'diSPIM-tilt-fast2':
            for i, view in enumerate(['SPIMA', 'SPIMB']):
                filename = folder + view + '/' + view + '_'+str(filenum)+'.tif'
                with tifffile.TiffFile(filename) as tf:
                    import time
                    start = time.time()
                    log.info('Reading '+filename)
                    data = tf.asarray() # ZPYX order
                    data = np.moveaxis(data, [1,0,2,3], [0,1,2,3]) # PZYX
                    if roi is not None: # Crop
                        data = data[:, roi[2][0]:roi[2][1], roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]]
                    if self.g.shape[0] != data.shape[3] and i == 0: # Make g with correct shape
                        datashape = (data.shape[3], data.shape[2], data.shape[1], 3, 2)
                        self.g = np.zeros(datashape, dtype=np.float32)
                    perc = np.percentile(data, bkg_perc)
                    print('Removing background: '+str(int(perc))) # In uint16 units
                    if data.dtype == np.uint16: # Convert 
                        data = (data/np.iinfo(np.uint16).max).astype(np.float32)
                    data -= np.percentile(data, bkg_perc) # in floats to avoid overflow
                    log.info('Swapping axes...')
                    self.g[:,:,:,:,i] = np.moveaxis(data, [3, 2, 1, 0], [0, 1, 2, 3]) # XYZPV order
                    del data

                    end = time.time()
                    print('File I/0 took '+str(np.round(end - start, 2))+' seconds.')
        
        elif format == 'diSPIM-tilt-fast':
            for i, view in enumerate(['SPIMA', 'SPIMB']):
                filename = folder + view + '/' + view + '_0.tif'
                with tifffile.TiffFile(filename) as tf:
                    import time
                    start = time.time()
                    log.info('Reading '+filename)
                    data = tf.asarray() # ZPYX order
                    data = data.reshape((3, data.shape[0]//3, data.shape[1], data.shape[2]))
                    data = np.moveaxis(data, [1,0,2,3], [0,1,2,3])
                    if roi is not None: # Crop
                        data = data[roi[2][0]:roi[2][1], :, roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]]
                    if self.g.shape[0] != data.shape[3]: # Make g with correct shape
                        datashape = (data.shape[3], data.shape[2], data.shape[0], 3, self.pols.shape[0])
                        self.g = np.zeros(datashape, dtype=np.float32)
                    if data.dtype == np.uint16: # Convert 
                        data = (data/np.iinfo(np.uint16).max).astype(np.float32)
                    log.info('Moving '+filename)
                    self.g[:,:,:,:,i] = np.moveaxis(data, [3, 2, 0, 1], [0, 1, 2, 3]) # XYZPTV order
                    del data

                    end = time.time()
                    print(end - start)
                    print('File I/0 took '+str(np.round(end - start, 2))+' seconds.')
        
        elif format == 'diSPIM-tilt':
            for i, view in enumerate(['SPIMA', 'SPIMB']):
                for j, tilt in enumerate(tilts):
                    filename = folder + view + '/' + view + '_Tilt_' + tilt + '.tif'
                    with tifffile.TiffFile(filename) as tf:
                        import time
                        start = time.time()

                        log.info('Reading '+filename)
                        data = tf.asarray() # ZPYX order
                        if roi is not None: # Crop
                            data = data[roi[2][0]:roi[2][1], :, roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]]
                        if self.g.shape[0] != data.shape[3]: # Make g with correct shape
                            datashape = (data.shape[3], data.shape[2], data.shape[0], len(tilts)*7, self.pols.shape[0])
                            self.g = np.zeros(datashape, dtype=np.float32)
                        if data.dtype == np.uint16: # Convert 
                            data = (data/np.iinfo(np.uint16).max).astype(np.float32)
                        log.info('Moving '+filename)
                        self.g[:,:,:,7*j:7*(j+1),i] = np.moveaxis(data, [3, 2, 0, 1], [0, 1, 2, 3]) # XYZPTV order
                        del data
                        
                        end = time.time()
                        print('File I/0 took '+str(np.round(end - start, 2))+' seconds.')

        elif format == 'diSPIM':
            for i, view in enumerate(['SPIMA', 'SPIMB']):                
                for j in range(3):
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
        else:
            for i, view in enumerate(['SPIMA', 'SPIMB']):
                filename = folder + view + '/' + view + '_reg_' + str(filenum) + '.tif'
                with tifffile.TiffFile(filename) as tf:
                    log.info('Reading '+filename)
                    data = tf.asarray() # ZPYX order
                    if roi is not None: # Crop
                        data = data[roi[2][0]:roi[2][1], :, roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]]
                    if self.g.shape[0] != data.shape[3]: # Make g with correct shape
                        datashape = (data.shape[3], data.shape[2], data.shape[0], self.pols.shape[1], self.pols.shape[0])
                        self.g = np.zeros(datashape, dtype=np.float32)
                    if data.dtype == np.uint16: # Convert 
                        data = (data/np.iinfo(np.uint16).max).astype(np.float32)
                    for j in range(datashape[3]):
                        if order is not None:
                            jj = order[i][j]
                        else:
                            jj = j
                        self.g[...,jj,i] = np.swapaxes(data[:,j,:,:], 0, 2) # XYZPV order

    def read_calibration(self, folder, XY_range=15):
        log.info('XY_range: ' + str(XY_range))
        self.g = np.zeros((2*XY_range, 2*XY_range, 30, 21, 2), dtype=np.float32)
        for i, view in enumerate(['SPIMA', 'SPIMB']):
            for j, tilt in enumerate(['0', '1', '-1']):
                filename = folder + view + '/' + view + '_Tilt_' + tilt + '_0.tif'
                with tifffile.TiffFile(filename) as tf:
                    import time
                    start = time.time()
                    log.info('Reading '+filename)
                    data = tf.asarray() # ZYX order
                    data = np.swapaxes(data, 0, 2) # XYZ
                    data = data.reshape(data.shape[0:2] + (7, 30)) # XYPZ

                    log.info('Moving and cropping '+filename)
                    X_mid = int(data.shape[0]/2)
                    Y_mid = int(data.shape[0]/2)
                    self.g[:,:,:,7*j:7*(j+1),i] = np.swapaxes(data, 2, 3)[X_mid-XY_range:X_mid+XY_range, Y_mid-XY_range:Y_mid+XY_range,...] # XYZPV

                    
    def plot_calibration_fit(self, out='fit.pdf'):
        f, axs = plt.subplots(1, 2, figsize=(10, 4))

        colors = ['g', 'b', 'r']
        labels = ['Tilt 0', 'Tilt 1', 'Tilt -1 ']

        offsets = []
        for j in range(2): # Views
            for k in range(3): # Tilts
                ax = axs[j]

                x0 = np.linspace(0,180,1000)
                x0t = np.array(np.deg2rad(x0))
                x1 = [0,45,60,90,120,135,180]
                x1t = np.array(np.deg2rad(x1))

                data_subset = self.g[...,7*k:7*(k+1),j]
                means = np.mean(data_subset, axis=(0,1,2))
                stds = np.std(data_subset, axis=(0,1,2))
                y = means # means

                # Least squares fit
                A = np.zeros((len(x1t), 3))
                A[:,0] = 1
                A[:,1] = np.cos(2*x1t)
                A[:,2] = np.sin(2*x1t)
                abc = np.linalg.lstsq(A, y, rcond=None)[0]

                def abc2theta(abc, theta):
                    return abc[0] + abc[1]*np.cos(2*theta) + abc[2]*np.sin(2*theta)
                y_lst = np.array([abc2theta(abc, np.deg2rad(xx)) for xx in x0])

                ax.errorbar(x1, y, yerr=stds, fmt='o'+colors[k], label=labels[k]) # Plot dots
                ax.plot(x0, y_lst, '-'+colors[k])

                means2 = np.mean(data_subset, axis=(0,1))
                xx = np.linspace(0,180,means2.flatten().shape[-1])
                ax.plot(xx, means2.T.flatten(), '-'+colors[k], alpha=0.3)

                offset = x0[np.argmax(y_lst)]
                offsets.append(offset)
                print('Pol Offset (deg): ' + '{:.2f}'.format(offset))

            # Labels
            ax.set_xlabel('Excitation polarization $\\hat{\\mathbf{p}}$ (degrees)')
            if j == 0:
                ax.set_ylabel('Counts')
            ax.set_xlim([-10,190])
            ax.set_ylim([0.9*np.min(np.mean(self.g, axis=(0,1,2))),
                         1.1*np.max(np.mean(self.g, axis=(0,1,2)))])
            ax.xaxis.set_ticks([0, 45, 90, 135, 180])
            ax.legend(frameon=False)

        axs[0].annotate('SPIMA', xy=(0,0), xytext=(0.5, 1.1), xycoords='axes fraction', textcoords='axes fraction', ha='center', va='center')
        axs[1].annotate('SPIMB', xy=(0,0), xytext=(0.5, 1.1), xycoords='axes fraction', textcoords='axes fraction', ha='center', va='center')
        
        f.savefig(out, bbox_inches='tight')

        return np.mean(offsets)

    def plot_data_means(self, out='mean.pdf'):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))

        view = ['A', 'B']
        pol = ['1', '2', '3']        
        
        for j in range(2): # Views
            for k in range(3): # Tilts
                means = np.mean(self.g, axis=(0,1))
                ax.plot(means[:,k,j], alpha=1, label=view[j]+pol[k])

            # Labels
            ax.set_xlabel('z slice')
            ax.set_ylabel('Mean intensity in slice (AU)')
            ax.legend(frameon=False)

        f.savefig(out, bbox_inches='tight')
