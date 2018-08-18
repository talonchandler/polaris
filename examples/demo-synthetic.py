# This script demonstrates the main functionality of polaris. It generates and
# visualizes a spatio-angular phantom (a Spang object), specifies a microscope
# (a MultiMicroscope object), generates data (a Data object) using the phantom
# and the microscope, and reconstructs the object using a pseudo-inverse
# solution.

from polaris import data, spang, phantom
from polaris.micro import multi
import numpy as np

# Make output folder
folder = './three-helix/'
import os
if not os.path.exists(folder):
    os.makedirs(folder)

# Generate phantom
vox_dim = (130,130,130) # nm
# helix phantom
px = (64,64,64)
phant = phantom.three_helix(vox_dim=vox_dim, px=px) 
# bead phantom - try varying orientation and kappa
# px = (32,32,32)
# phant = phantom.bead(orientation=[1,0,0], kappa=3, vox_dim=vox_dim, px=px)

# Calculate phantom statistics and save
phant.calc_stats()
# masking is necessary for fast rendering
mask = phant.density > 0.1 
# try "interact=True" to interact with the phantom
phant.visualize(folder+'phantom/', mask=mask, interact=False, video=True,
                n_frames=15) 
phant.save_stats(folder+'phantom/')
phant.save_summary(folder+'phantom.pdf', mask=mask)

# Specify microscope
data1 = data.Data(g=np.zeros(phant.f.shape[0:3]+(4,2)), vox_dim=vox_dim,
                  det_nas=[1.1, 0.71])
m = multi.MultiMicroscope(phant, data1, n_samp=1.33, lamb=525,
                          spang_coupling=True)

# Calculate system matrix
m.calc_H()

# Generate data using forward model
# set "snr" to a positive number to simulate Poisson noise 
data1.g = m.fwd(phant.f, snr=None) 

# Save data
data1.save_mips(folder+'data.pdf')
data1.save_tiff(folder+'data/')

# Calculate pseudoinverse solution
# set "eta" to a positive number for Tikhonov regularization
phant.f = m.pinv(data1.g, eta=0)

# Calculate reconstruction statistics and save
phant.calc_stats()
mask = phant.density > 0.1
phant.visualize(folder+'phantom-recon/', mask=mask, interact=False, video=True,
                n_frames=15)
phant.save_stats(folder+'phantom-recon/')
phant.save_summary(folder+'phantom-recon.pdf', mask=mask)
