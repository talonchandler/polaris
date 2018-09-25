from polaris import spang, phantom, data
from polaris.micro import multi
import numpy as np

# Specify input and output
in_folder = '/Users/Talon/local-data/2018-dispim-min/20180725_Polimaging_FixedCells_eGFP-210_R-Ph/Cell4_Processed/Registered/561/'
out_folder = './roi2-561/'

# in_folder = '/Users/Talon/local-data/2018-dispim-min/20180725_Polimaging_FixedCells_eGFP-210_R-Ph/Cell4_Processed/Registered/488/'
# out_folder = './roi2-488/'

# Specify voxel dimensions in nm
vox_dim_data = [130, 130, 130]
vox_dim_A = [130, 130, 549]
vox_dim_B = [227, 227, 835.8]

# 561 calibration data
# Manually measured from ROIs in fluorescent lake data in P0-P4 order
# /Volumes/lacie-win/2018-07-30-pol-dipsim 
epi_cal = [[8150, 8550, 8650, 8350], [16250, 16750, 17000, 16500]]
ls_cal = [[6050, 5200, 4900, 5800], [9570, 7350, 6700, 8800]]

# Read data into memory
order = [[2, 1, 0, 3], [2, 3, 0, 1]] # For reordering data
data1 = data.Data(vox_dim=vox_dim_data, det_nas=[1.1, 0.71])

# First pass
# data1.read_tiff(in_folder, order=order, roi=None)
# data1.save_mips(out_folder + 'data-raw.pdf', normalize=True)

# Second pass
# roi = [[100, 300], [50, 300], [100, 250]]
# roi = [[0, 400], [0, 400], [0, 400]]
# data1.read_tiff(in_folder, order=order, roi=roi)
# data1.save_tiff(out_folder+'data-roi/', diSPIM_format=True)
# data1.save_mips(out_folder + 'data-roi.pdf', normalize=True)

# Third pass
data1.read_tiff(out_folder+'data-roi/')

# Make spang and microscope objects
spang1 = spang.Spang(f=np.zeros(data1.g.shape[0:3] + (15,)),
                     vox_dim=vox_dim_data)
micro1 = multi.MultiMicroscope(spang1, data1, n_samp=1.33, lamb=525,
                               spang_coupling=True)

# Calculate expected values from fluorescent lake
lake_response = micro1.lake_response(data1)

# Apply corrections
data1.remove_background()
# data1.apply_calibration_correction(epi_cal, ls_cal, order, lake_response)
data1.apply_power_correction()
data1.apply_voxel_size_correction(vox_dim_A, vox_dim_B)
# data1.apply_padding()

# Save corrected data
# data1.save_mips(out_folder + 'data-corrected.pdf', normalize=True)
# data1.save_tiff(out_folder+'data-corrected/', diSPIM_format=True)

# Calculate system matrix
# micro1.calc_H(); micro1.save_H(out_folder + 'H.npz')
micro1.load_H(out_folder + 'H.npz')

# Calculate pseudoinverse solution
# set "eta" to a list of positive number for Tikhonov regularization
eta = 1e0
eta_string = '{:.0e}'.format(eta)
print("Reconstructing with eta = " + eta_string)
spang1.f = micro1.pinv(data1.g, eta=eta)

import pdb; pdb.set_trace() 
spang1.save_stats(out_folder+'recon-' + eta_string + '/')
# spang1.visualize('./ellipsoid/', mask=spang1.density() > 0.04, interact=True, video=False, n_frames=1, skip_n=6, scale=5, viz_type='ELLIPSOID')
# spang1.visualize('./ellipsoid/', mask=spang1.density() > 0.04, interact=False, video=True, n_frames=360, skip_n=6, scale=5, viz_type='ELLIPSOID')
spang1.visualize(out_folder+'ellipsoid/', mask=spang1.density() > 0.04, interact=True, video=False, n_frames=1, skip_n=10, scale=7, viz_type='ELLIPSOID')
spang1.save_summary(out_folder+'recon-' + eta_string + '.pdf', density_filter=0.04, skip_n=6, scale=5)
                   
