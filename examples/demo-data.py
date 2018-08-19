# This script demonstrates a spatio-angular reconstruction using polaris. It
# reads the files in guv-data/ and outputs summaries of the data (data.pdf),
# and the reconstructed object (recon.pdf, recon.avi) to guv-recon/. 

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
order = [[2, 1, 0, 3], [2, 3, 0, 1]]
data1.read_tiff('./guv-data/', order=order)

# Apply corrections
# Adjust for pixel sizes 2 powers for input power, 3 powers for voxel size
data1.g[...,0] = (1.1/0.71)**5*data1.g[...,0]
# TODO: Calibration correction
# TODO: Background correction
data1.save_mips(folder + 'data.pdf', normalize=True)

# Build microscope model
spang1 = spang.Spang(f=np.zeros(data1.g.shape[0:3] + (15,)), vox_dim=vox_dim)
m = multi.MultiMicroscope(spang1, data1, n_samp=1.33, lamb=525,
                          spang_coupling=True)

# Calculate system matrix
m.calc_H()
# m.save_H(folder + 'H.npz') # try saving and loading H to save time
# m.load_H(folder + 'H.npz')

# Calculate pseudoinverse solution
# set "eta" to a positive number for Tikhonov regularization
spang1.f = m.pinv(data1.g, eta=1e0)

# Calculate reconstruction statistics and save
spang1.calc_stats()
mask = spang1.density > 0.6
spang1.visualize(folder+'guv-recon/', mask=mask, interact=False, video=True,
                 n_frames=15, skip_n=3)
spang1.save_stats(folder+'guv-recon/')
spang1.save_summary(folder+'guv-recon.pdf', mask=mask, skip_n=3)
