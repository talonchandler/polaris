import numpy as np
from tqdm import tqdm
import logging
from joblib import Parallel, delayed
from polaris.evaluation import eval
import os

log = logging.getLogger('log')


class recon_single:
    def __init__(self, multi):
        self.dispim = multi
        self.H = multi.Hxyz

        self.gaunt = multi.Gaunt * 3.5449077
        self.s = multi.data.g.shape[0:3]

        self.calc_H()

    def calc_H(self):
        log.info('Computing H_back')

        self.H_back = self.H.conjugate()

        sv = self.H.sum(axis=(0, 1, 2, 4))

        for p in range(self.H.shape[4]):
            self.H_back[..., p] = self.SHDiv_1D(self.H_back[..., p], sv)

    def ConvFFT3(self, Vol, OTF, order):
        Vol_fft = np.fft.rfftn(Vol, axes=(0, 1, 2))
        temp = []
        if order == 0:
            temp = np.zeros(OTF.shape[0:3] + (OTF.shape[4],), dtype=np.complex64)
            Parallel(n_jobs=-1, backend='threading')(
                [delayed(self.compute_ConvFFT3)(temp, Vol_fft, OTF, z, 0) for z in range(temp.shape[2])])
        if order == 1:
            temp = np.zeros(OTF.shape[0:3] + (OTF.shape[3],), dtype=np.complex64)
            Parallel(n_jobs=-1, backend='threading')(
                [delayed(self.compute_ConvFFT3)(temp, Vol_fft, OTF, z, 1) for z in range(temp.shape[2])])
        if order == 2:
            temp = np.zeros(OTF.shape[0:3] + (OTF.shape[3],), dtype=np.complex64)
            Parallel(n_jobs=-1, backend='threading')(
                [delayed(self.compute_ConvFFT3)(temp, Vol_fft, OTF, z, 2) for z in range(temp.shape[2])])
        Vol = np.real(np.fft.irfftn(temp, s=Vol.shape[0:3], axes=(0, 1, 2)))
        return Vol

    def compute_ConvFFT3(self, temp, inVol_fft, OTF, z, order):
        if order == 0:
            temp[:, :, z, :] = np.einsum('xyj,xyjp->xyp', inVol_fft[:, :, z, :], OTF[:, :, z, :, :])
        if order == 1:
            temp[:, :, z, :] = np.einsum('xyp,xyjp->xyj', inVol_fft[:, :, z, :], OTF[:, :, z, :, :])
        if order == 2:
            temp[:, :, z, :] = np.einsum('xyjs,xys->xyj', OTF[:, :, z, :, :], inVol_fft[:, :, z, :])

    def SHMul(self, SH0, SH1):
        outSH = SH0.copy() * 0
        Parallel(n_jobs=-1, backend='threading')(
            [delayed(self.compute_SHMul)(outSH, SH0, SH1, z) for z in range(SH0.shape[2])])
        return outSH

    def compute_SHMul(self, outSH, SH0, SH1, z):
        mat = np.einsum('jls,xys->xyjl', self.gaunt, SH0[:, :, z, :])
        outSH[:, :, z, :] = np.einsum('xyjl,xyl->xyj', mat, SH1[:, :, z, :])

    def SHDiv(self, SH0, SH1):
        outSH = SH0.copy() * 0
        Parallel(n_jobs=-1, backend='threading')(
            [delayed(self.compute_SHDiv)(outSH, SH0, SH1, z) for z in range(SH0.shape[2])])
        return outSH

    def compute_SHDiv(self, outSH, SH0, SH1, z):
        mat = np.einsum('jls,xys->xyjl', self.gaunt, SH1[:, :, z, :])
        mat_inv = np.linalg.inv(mat)
        outSH[:, :, z, :] = np.einsum('xyjl,xyl->xyj', mat_inv, SH0[:, :, z, :])

    def SHDiv_1D(self, SH0, SH1):
        mat = np.einsum('jls,s->jl', self.gaunt, SH1)
        mat_inv = np.linalg.inv(mat)
        outSH = np.einsum('jl,xyzl->xyzj', mat_inv, SH0)
        return outSH

    def recon(self, g, iter_num=10):
        log.info('Applying Richardson-Lucy recon')

        img = (g - g.min()) / (g.max() - g.min())
        img = img.reshape(self.s + (-1,))

        ek = np.zeros(self.s + (15,))
        ek[..., 0] = 1

        for iter in tqdm(range(iter_num)):
            fwd = self.ConvFFT3(ek, self.H, order=0)
            fwd[fwd < 1e-10] = 1e-10
            dif = img / fwd
            del fwd
            bwd = self.ConvFFT3(dif, self.H_back, order=1)
            del dif
            ek = self.SHMul(ek, bwd)

        return ek

    def recon_loss(self, g, iter_num, phant):
        img = (g - g.min()) / (g.max() - g.min())
        img = img.reshape(self.s + (-1,))

        ek = np.zeros(self.s + (15,))
        ek[..., 0] = 1

        ssim_rcd = np.zeros(iter_num + 1)
        peak_rcd = np.zeros(iter_num + 1)

        label_f = phant.f
        BinvT, Bvertices = phant.Binv.T, phant.sphere.vertices

        ssim_rcd[0] = eval.SSIM(ek[..., 0], phant.f[..., 0])
        peak_rcd[0] = eval.PeakDif(ek, label_f, BinvT, Bvertices)

        for iter in tqdm(range(iter_num)):
            fwd = self.ConvFFT3(ek, self.H, order=0)
            fwd[fwd < 1e-10] = 1e-10
            dif = img / fwd
            del fwd
            bwd = self.ConvFFT3(dif, self.H_back, order=1)
            del dif
            ek = self.SHMul(ek, bwd)

            ssim_rcd[iter + 1] = eval.SSIM(ek[..., 0], phant.f[..., 0])
            peak_rcd[iter + 1] = eval.PeakDif(ek, label_f, BinvT, Bvertices)

        return ssim_rcd, peak_rcd


class recon_dual:
    def __init__(self, multi):
        self.dispim = multi

        self.Ha = multi.H0
        self.Hb = multi.H1

        self.gaunt = multi.Gaunt * 3.5449077
        self.s = multi.data.g.shape[0:3]

        self.calc_H()

    def calc_H(self):
        log.info('Computing H_back')

        self.Ha_back = self.Ha.conjugate()
        self.Hb_back = self.Hb.conjugate()

        sva = self.Ha.sum(axis=(0, 1, 2, 4))
        svb = self.Hb.sum(axis=(0, 1, 2, 4))

        for p in range(self.Ha.shape[4]):
            self.Ha_back[..., p] = self.SHDiv_1D(self.Ha_back[..., p], sva)
            self.Hb_back[..., p] = self.SHDiv_1D(self.Hb_back[..., p], svb)

    def ConvFFT3(self, Vol, OTF, order):
        Vol_fft = np.fft.rfftn(Vol, axes=(0, 1, 2))
        temp = []
        if order == 0:
            temp = np.zeros(OTF.shape[0:3] + (OTF.shape[4],), dtype=np.complex64)
            Parallel(n_jobs=-1, backend='threading')(
                [delayed(self.compute_ConvFFT3)(temp, Vol_fft, OTF, z, 0) for z in range(temp.shape[2])])
        if order == 1:
            temp = np.zeros(OTF.shape[0:3] + (OTF.shape[3],), dtype=np.complex64)
            Parallel(n_jobs=-1, backend='threading')(
                [delayed(self.compute_ConvFFT3)(temp, Vol_fft, OTF, z, 1) for z in range(temp.shape[2])])
        if order == 2:
            temp = np.zeros(OTF.shape[0:3] + (OTF.shape[3],), dtype=np.complex64)
            Parallel(n_jobs=-1, backend='threading')(
                [delayed(self.compute_ConvFFT3)(temp, Vol_fft, OTF, z, 2) for z in range(temp.shape[2])])
        Vol = np.real(np.fft.irfftn(temp, s=Vol.shape[0:3], axes=(0, 1, 2)))
        return Vol

    def compute_ConvFFT3(self, temp, inVol_fft, OTF, z, order):
        if order == 0:
            temp[:, :, z, :] = np.einsum('xyj,xyjp->xyp', inVol_fft[:, :, z, :], OTF[:, :, z, :, :])
        if order == 1:
            temp[:, :, z, :] = np.einsum('xyp,xyjp->xyj', inVol_fft[:, :, z, :], OTF[:, :, z, :, :])
        if order == 2:
            temp[:, :, z, :] = np.einsum('xyjs,xys->xyj', OTF[:, :, z, :, :], inVol_fft[:, :, z, :])

    def SHMul(self, SH0, SH1):
        outSH = SH0.copy() * 0
        Parallel(n_jobs=-1, backend='threading')(
            [delayed(self.compute_SHMul)(outSH, SH0, SH1, z) for z in range(SH0.shape[2])])
        return outSH

    def compute_SHMul(self, outSH, SH0, SH1, z):
        mat = np.einsum('jls,xys->xyjl', self.gaunt, SH0[:, :, z, :])
        outSH[:, :, z, :] = np.einsum('xyjl,xyl->xyj', mat, SH1[:, :, z, :])

    def SHDiv(self, SH0, SH1):
        outSH = SH0.copy() * 0
        Parallel(n_jobs=-1, backend='threading')(
            [delayed(self.compute_SHDiv)(outSH, SH0, SH1, z) for z in range(SH0.shape[2])])
        return outSH

    def compute_SHDiv(self, outSH, SH0, SH1, z):
        mat = np.einsum('jls,xys->xyjl', self.gaunt, SH1[:, :, z, :])
        mat_inv = np.linalg.inv(mat)
        outSH[:, :, z, :] = np.einsum('xyjl,xyl->xyj', mat_inv, SH0[:, :, z, :])

    def SHDiv_1D(self, SH0, SH1):
        mat = np.einsum('jls,s->jl', self.gaunt, SH1)
        mat_inv = np.linalg.inv(mat)
        outSH = np.einsum('jl,xyzl->xyzj', mat_inv, SH0)
        return outSH

    def recon(self, g, iter_num=10, mod=0):
        log.info('Applying Richardson-Lucy recon')

        img = (g - g.min()) / (g.max() - g.min())
        imga, imgb = img[..., 0], img[..., 1]

        ek = np.zeros(g.shape[0:3] + (15,))
        ek[..., 0] = 1

        if mod == 0:
            for iter in tqdm(range(iter_num)):
                fwd = self.ConvFFT3(ek, self.Ha, order=0)
                fwd[fwd < 1e-10] = 1e-10
                dif = imga / fwd
                del fwd
                bwd = self.ConvFFT3(dif, self.Ha_back, order=1)
                del dif
                ek = self.SHMul(ek, bwd)

                fwd = self.ConvFFT3(ek, self.Hb, order=0)
                fwd[fwd < 1e-10] = 1e-10
                dif = imgb / fwd
                del fwd
                bwd = self.ConvFFT3(dif, self.Hb_back, order=1)
                del dif
                ek = self.SHMul(ek, bwd)

        if mod == 1:
            for iter in tqdm(range(iter_num)):
                fwd = self.ConvFFT3(ek, self.Ha, order=0)
                fwd[fwd < 1e-10] = 1e-10
                dif = imga / fwd
                del fwd
                bwd = self.ConvFFT3(dif, self.Ha_back, order=1)
                del dif
                ek_a = self.SHMul(ek, bwd)

                fwd = self.ConvFFT3(ek, self.Hb, order=0)
                fwd[fwd < 1e-10] = 1e-10
                dif = imgb / fwd
                del fwd
                bwd = self.ConvFFT3(dif, self.Hb_back, order=1)
                del dif
                ek_b = self.SHMul(ek, bwd)

                ek = (ek_a + ek_b) / 2

        return ek

    def recon_loss(self, g, iter_num, phant, mod=0):
        img = (g - g.min()) / (g.max() - g.min())
        imga, imgb = img[..., 0], img[..., 1]

        ek = np.zeros(g.shape[0:3] + (15,))
        ek[..., 0] = 1

        ssim_rcd = np.zeros(iter_num + 1)
        peak_rcd = np.zeros(iter_num + 1)

        label_f = phant.f
        BinvT, Bvertices = phant.Binv.T, phant.sphere.vertices

        ssim_rcd[0] = eval.SSIM(ek[..., 0], phant.f[..., 0])
        peak_rcd[0] = eval.PeakDif(ek, label_f, BinvT, Bvertices)

        if mod == 0:
            for iter in tqdm(range(iter_num)):
                fwd = self.ConvFFT3(ek, self.Ha, order=0)
                fwd[fwd < 1e-10] = 1e-10
                dif = imga / fwd
                del fwd
                bwd = self.ConvFFT3(dif, self.Ha_back, order=1)
                del dif
                ek = self.SHMul(ek, bwd)

                fwd = self.ConvFFT3(ek, self.Hb, order=0)
                fwd[fwd < 1e-10] = 1e-10
                dif = imgb / fwd
                del fwd
                bwd = self.ConvFFT3(dif, self.Hb_back, order=1)
                del dif
                ek = self.SHMul(ek, bwd)

                ssim_rcd[iter + 1] = eval.SSIM(ek[..., 0], phant.f[..., 0])
                peak_rcd[iter + 1] = eval.PeakDif(ek, label_f, BinvT, Bvertices)

        if mod == 1:
            for iter in tqdm(range(iter_num)):
                fwd = self.ConvFFT3(ek, self.Ha, order=0)
                fwd[fwd < 1e-10] = 1e-10
                dif = imga / fwd
                del fwd
                bwd = self.ConvFFT3(dif, self.Ha_back, order=1)
                del dif
                ek_a = self.SHMul(ek, bwd)

                fwd = self.ConvFFT3(ek, self.Hb, order=0)
                fwd[fwd < 1e-10] = 1e-10
                dif = imgb / fwd
                del fwd
                bwd = self.ConvFFT3(dif, self.Hb_back, order=1)
                del dif
                ek_b = self.SHMul(ek, bwd)

                ek = (ek_a + ek_b) / 2

                ssim_rcd[iter + 1] = eval.SSIM(ek[..., 0], phant.f[..., 0])
                peak_rcd[iter + 1] = eval.PeakDif(ek, label_f, BinvT, Bvertices)

        return ssim_rcd, peak_rcd
