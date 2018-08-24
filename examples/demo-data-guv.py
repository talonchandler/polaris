# This script demonstrates a spatio-angular reconstruction using polaris. It
# reads files in guv-data/ and outputs summaries of the data with correction
# factors applied (data-corrected.pdf), and the reconstructed object (recon.pdf,
# recon.avi) to guv-recon/.

from polaris import spang, phantom, data
from polaris.micro import multi
import numpy as np

# Make output folder
folder = './guv-recon/'
import os
if not os.path.exists(folder):
    os.makedirs(folder)

# Read data and save mips
vox_dim = (130,130,130) # nm
data1 = data.Data(vox_dim=vox_dim, det_nas=[1.1, 0.71])
order = np.array([[2, 1, 0, 3], [2, 3, 0, 1]])
data1.read_tiff('./guv-data/', order=order)

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
# Adjust for pixel sizes 2 powers for input power, 3 powers for voxel size
data1.g[...,0] = (1.1/0.71)**5*data1.g[...,0]
data1.save_mips(folder + 'data-corrected.pdf', normalize=True)
data1.save_tiff(folder+'data-corrected/', diSPIM_format=True) 

# Calculate system matrix
m.calc_H()
# m.save_H(folder + 'H.npz') # try saving and loading H to save time
# m.load_H(folder + 'H.npz')

# Calculate pseudoinverse solution
# set "eta" to a positive number for Tikhonov regularization
spang1.f = m.pinv(data1.g, eta=1e0)

# Calculate reconstruction statistics and save
spang1.calc_stats()
mask = spang1.density > 0.5
spang1.visualize(folder+'guv-recon/', mask=mask, interact=False, video=True,
                 n_frames=15, skip_n=3)
spang1.save_stats(folder+'guv-recon/')
spang1.save_summary(folder+'guv-recon.pdf', mask=mask, skip_n=3)
