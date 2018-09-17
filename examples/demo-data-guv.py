# This script demonstrates a spatio-angular reconstruction using polaris. It
# reads files in guv-data/ and outputs summaries of the data with correction
# factors applied (data-corrected.pdf), and the reconstructed object (recon.pdf,
# recon.avi) to guv-recon/.

from polaris import spang, phantom, data
from polaris.micro import multi
import numpy as np

# Specify input and output
in_folder = './guv-data/' 
out_folder = './guv-recon/'
import os
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# Read data and save mips
vox_dim = (130,130,130) # nm
data1 = data.Data(vox_dim=vox_dim, det_nas=[1.1, 0.71])
order = np.array([[2, 1, 0, 3], [2, 3, 0, 1]])
data1.read_tiff(in_folder, order=order)

# Model for uniform distributions
# Build microscope model
spang1 = spang.Spang(f=np.zeros(data1.g.shape[0:3] + (15,)), vox_dim=vox_dim)
m = multi.MultiMicroscope(spang1, data1, n_samp=1.33, lamb=525,
                          spang_coupling=True)

# Calculate expected values from fluorescent lake
e0 = m.calc_point_H(0,0,0,0,data1.pols_norm)[0,:]
e1 = m.calc_point_H(0,0,0,1,data1.pols_norm)[0,:]
expected = np.vstack([e0, e1])

# Measured values from fluorescent lake 
# 561 calibration data. Values measured by hand from unifrom ROIs in
# /Volumes/lacie-win/2018-07-30-pol-dipsim in P0-P4 order (no swaps)
epi_cal = np.array([[8150,8550,8650,8350], [16250,16750,17000,16500]])
ls_cal = np.array([[6050,5200,4900,5800], [9570,7350,6700,8800]])
epi_normed = epi_cal/np.mean(epi_cal, axis=-1)[:,None]
ls_corrected = ls_cal/epi_normed
cal_data = ls_corrected/np.mean(ls_corrected, axis=-1)[:,None]
cal_data[0,:] = cal_data[0,order[0,:]] # reorder
cal_data[1,:] = cal_data[1,order[1,:]] # reorder

# Apply corrections
data1.remove_background()
data1.apply_calibration_correction(cal_data, expected)

# Adjust input power
data1.g[...,0] = (1.1/0.71)**2*data1.g[...,0]

# Correct voxel sizes
Avox = np.array([0.13, 0.13, 0.549])
Bvox = np.array([0.227, 0.227, 0.8358])
vox_factor = np.prod(Bvox/Avox)
data1.g[...,0] = vox_factor*data1.g[...,0]

# Save corrected data
data1.save_mips(out_folder + 'data-corrected.pdf', normalize=True)
data1.save_tiff(out_folder+'data-corrected/', diSPIM_format=True) 

# Calculate system matrix
m.calc_H()
# m.save_H(folder + 'H.npz') # try saving and loading H to save time
# m.load_H(folder + 'H.npz')

# Calculate pseudoinverse solution
# set "etas" to a list of positive number for Tikhonov regularization
etas = [1e-1]
for eta in etas:
    eta_string = '{:.1e}'.format(eta)
    print("Reconstructing with eta = " + eta_string)
    
    spang1.f = m.pinv(data1.g, eta=eta)

    # Calculate reconstruction statistics and save
    density_filter = 0.2
    # spang1.visualize(out_folder+'guv-recon-' + eta_string + '/',
    #                  mask=spang1.density() > density_filter, interact=False,
    #                  video=True, n_frames=15, skip_n=7, viz_type='PEAK')
    spang1.save_stats(out_folder+'guv-recon/')
    spang1.save_summary(out_folder+'guv-recon.pdf',
                        density_filter=density_filter, skip_n=7)
