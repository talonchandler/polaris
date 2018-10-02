def main():
    # This script demonstrates the main functionality of polaris. It generates and
    # visualizes a spatio-angular phantom (a Spang object), specifies a microscope
    # (a MultiMicroscope object), generates data (a Data object) using the phantom
    # and the microscope, and reconstructs the object using a pseudo-inverse
    # solution.

    from polaris import data, spang, phantom
    from polaris.micro import multi
    import numpy as np

    # Make output folder
    folder = './helix/'
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Generate phantom
    vox_dim = (130,130,130) # nm
    # helix phantom
    px = (64,64,64)
    # px = (64,128,64)
    # px = (64,256,128)
    # px = (64,128,256)
    phant = phantom.three_helix(vox_dim=vox_dim, px=px)
    # bead phantom - try varying orientation and kappa
    # px = (32,32,32)
    # phant = phantom.bead(orientation=[1,0,0], kappa=30, vox_dim=vox_dim, px=px)

    # Mask for fast rendering
    mask = phant.density() > 0
    # uncomment and try "interact=True" to interact with the phantom
    phant.visualize(folder+'phantom/', mask=mask, interact=False, video=True,
                    n_frames=360, scale=3,
                    viz_type=['ODF','Peak','Density'], tiff=True,
                    mag=4, skip_n=2)
                    
    # phant.save_stats(folder+'phantom/')
    phant.save_summary(folder+'phantom.pdf', mask=mask)

    # Specify microscope
    # try "det_nas = [0.8, 0.8]" for symmetric diSPIM
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
    # data1.save_mips(folder+'data.pdf')
    # try "diSPIM_format=False" for a 5D ImageJ hyperstack (pol = c, view = t)
    data1.save_tiff(folder+'data/', diSPIM_format=True) 

    # Calculate pseudoinverse solution
    # set "eta" to a positive number for Tikhonov regularization
    phant.f = m.pinv(data1.g, eta=0)

    # Calculate reconstruction statistics and save
    mask = phant.density() > 0.1
    phant.visualize(folder+'phantom-recon/', mask=mask, interact=False, video=True,
                    n_frames=36, viz_type=['ODF','Peak', 'Density'],
                    tiff=True, skip_n=2, scale=3)
    # phant.save_stats(folder+'phantom-recon/')
    phant.save_summary(folder+'phantom-recon.pdf', mask=mask)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
