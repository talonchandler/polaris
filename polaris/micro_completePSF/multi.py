# Complete PSF
from polaris.micro_completePSF import ill, det, micro
import numpy as np
from tqdm import tqdm
import logging
import os

log = logging.getLogger('log')


class MultiMicroscope:
    def __init__(self, spang, data, FWHM=2000, n_samp=1.33, lamb=525):
        self.spang = spang
        self.data = data
        self.X = spang.X
        self.Y = spang.Y
        self.Z = spang.Z
        self.J = spang.J
        self.P = data.P
        self.V = data.V
        self.lamb = lamb
        self.FWHM = FWHM
        self.n_samp = n_samp

        m = []
        for i, det_optical_axis in enumerate(data.det_optical_axes):
            ill_ = ill.Illuminator(data=data, optical_axis=data.ill_optical_axes[i])
            det_ = det.Detector(spang=spang, data=data, optical_axis=data.det_optical_axes[i],
                                lamb=lamb, na=data.det_nas[i], n=n_samp, FWHM=FWHM)
            m.append(micro.Microscope(spang=spang, data=data, ill=ill_, det=det_))  # Add microscope
        self.micros = m

        self.Gaunt_633 = np.load(os.path.join(os.path.dirname(__file__),
                                              '../harmonics/gaunt_633.npy'))
        self.Gaunt = np.load(os.path.join(os.path.dirname(__file__), '../harmonics/gaunt_l4.npy'))

    def calc_H(self):
        log.info('Computing H for view 0')
        self.H0 = self.micros[0].calc_H()

        log.info('Computing H for view 1')
        self.H1 = self.micros[1].calc_H()

        self.Hxyz = np.stack([self.H0, self.H1], axis=-1)
        self.Hxyz = np.reshape(self.Hxyz, self.H0.shape[0:3] + (15, self.P * self.V,))

    def pinv(self, g, eta):
        log.info('Applying pseudoinverse operator')

        G = np.fft.rfftn(g, axes=(0, 1, 2))
        G2 = np.reshape(G, G.shape[0:3] + (self.P * self.V,))

        from joblib import Parallel, delayed
        F = np.zeros(self.Hxyz.shape[0:3] + (self.J,), dtype=np.complex64)
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(self.compute_pinv)(F, G2, z, eta) for z in range(self.Hxyz.shape[2])]))

        del G2, G
        f = np.fft.irfftn(F, s=g.shape[0:3], axes=(0, 1, 2))
        return np.real(f)

    def compute_pinv(self, F, G2, z, eta):
        u, s, vh = np.linalg.svd(self.Hxyz[:, :, z, :], full_matrices=False)
        sreg = np.where(s > 1e-7, s / (s ** 2 + eta), 0)
        Pinv = np.einsum('xysv,xyv,xyvd->xysd', u, sreg, vh)
        F[:, :, z, :] = np.einsum('xysd,xyd->xys', Pinv, G2[:, :, z, :])

    def fwd(self, f, snr=None):
        log.info('Applying forward operator')

        # 3D FT
        F = np.fft.rfftn(f, axes=(0, 1, 2))

        # Tensor multiplication
        from joblib import Parallel, delayed
        G2 = np.zeros(self.Hxyz.shape[0:3] + (self.Hxyz.shape[4],), dtype=np.complex64)
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(self.compute_fwd)(G2, F, z) for z in range(self.Hxyz.shape[2])]))
        G = np.reshape(G2, G2.shape[0:3] + (self.P,) + (self.V,))

        # 3D IFT
        g = np.real(np.fft.irfftn(G, s=f.shape[0:3], axes=(0, 1, 2)))

        # Apply Poisson noise
        if snr is not None:
            norm = snr ** 2 / np.max(g)
            g = g * norm
            lam = (g.mean() / snr) ** 2
            noise = np.random.poisson(lam=lam, size=g.shape) - lam
            g = (g + noise) / norm
            g = np.abs(g)

        g = g / np.max(g)
        return g

    def compute_fwd(self, G2, F, z):
        G2[:, :, z, :] = np.einsum('xysp,xys->xyp', self.Hxyz[:, :, z, :, :], F[:, :, z, :])

    def save_H(self, filename):
        np.save(filename, self.Hxyz)

    def load_H(self, filename):
        self.Hxyz = np.load(filename)
