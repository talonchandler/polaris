from polaris import spang, data, util, phantom
from polaris.micro_completePSF import multi as multi_completePSF
from polaris.recon import recon_ISRA, recon_RL
import numpy as np
import os

if __name__ == '__main__':
    out_folder = './phantom/'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Generate phantom
    vox_dim = (130, 130, 130)
    px = (64, 64, 64)

    phant = phantom.three_helix(vox_dim=vox_dim, px=px)

    di0 = np.arange(21)
    di1 = np.arange(21)

    pols = util.pols_from_tilt(di0, di1, pol_offset=0, t0A=0.2, t0B=0.2, t1A=9.2, t1B=4.8, t_1A=-5.3, t_1B=-6.5)

    data1 = data.Data(g=np.zeros(phant.f.shape[0:3] + (4, 2)), vox_dim=vox_dim, det_nas=[1.1, 0.67], pols=pols)
    spang1 = spang.Spang(f=np.zeros(data1.g.shape[0:3] + (15,)), vox_dim=vox_dim)

    # Choose the model: CompletePSF
    ms = multi_completePSF.MultiMicroscope(phant, data1, FWHM=2000, n_samp=1.33, lamb=525)
    ms.calc_H()

    data1.g = ms.fwd(phant.f, snr=None)

    # Choose RL or ISRA, and Single-View or Dual-View
    Iter = recon_RL.recon_single(ms)
    # Iter = recon_RL.recon_dual(ms)
    # Iter = recon_ISRA.recon_single(ms)
    # Iter = recon_ISRA.recon_dual(ms)

    # If choose ISRA/RL -SingleView
    spang1.f = Iter.recon(data1.g, iter_num=10)

    # If choose ISRA/RL -DualView
    # spang1.f = Iter.recon(data1.g, iter_num=10, mod=0)  # mod == 0: alternating deconvolution
    # spang1.f = Iter.recon(data1.g, iter_num=10, mod=1) # mod == 1: additive deconvolution

    spang1.visualize(out_folder + 'recon/', mask=spang1.density() > 0.2, interact=True, video=False, n_frames=18,
                     viz_type=['Peak'], skip_n=2, scale=3)
