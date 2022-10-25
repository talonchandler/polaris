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
        log.info('Computing H_back and H_con')

        self.H_back = self.H.conjugate()

        self.H_con = np.zeros((self.H.shape[0:3]) + (15, 15,), dtype=np.complex64)
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(self.compute_H_con)(z) for z in range(self.H_con.shape[2])]))

        del self.H

    def compute_H_con(self, z):
        self.H_con[:, :, z, :, :] = np.einsum('xyjp,xysp->xyjs', self.H_back[:, :, z, :, :], self.H[:, :, z, :, :])

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

    def recon(self, g, iter_num=10):
        log.info('Applying ISRA recon')

        img = (g - g.min()) / (g.max() - g.min())
        img = img.reshape(self.s + (-1,))

        ek = np.zeros(self.s + (15,))
        ek[..., 0] = 1

        mid = self.ConvFFT3(img, self.H_back, order=1)

        for iter in tqdm(range(iter_num)):
            bwd = self.ConvFFT3(ek, self.H_con, order=2)
            dif = self.SHDiv(mid, bwd)
            del bwd
            ek = self.SHMul(ek, dif)
            del dif

        return ek

    def recon_loss(self, g, iter_num, phant):

        g = (g - g.min()) / (g.max() - g.min())
        img = g.reshape(self.s + (-1,))

        ek = np.zeros(self.s + (15,))
        ek[..., 0] = 1

        mid = self.ConvFFT3(img, self.H_back, order=1)

        ssim_rcd = np.zeros(iter_num + 1)
        peak_rcd = np.zeros(iter_num + 1)

        label_f = phant.f
        BinvT, Bvertices = phant.Binv.T, phant.sphere.vertices

        ssim_rcd[0] = eval.SSIM(ek[..., 0], phant.f[..., 0])
        peak_rcd[0] = eval.PeakDif(ek, label_f, BinvT, Bvertices)

        for iter in range(iter_num):
            bwd = self.ConvFFT3(ek, self.H_con, order=2)
            dif = self.SHDiv(mid, bwd)
            del bwd
            ek = self.SHMul(ek, dif)
            del dif

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
        log.info('Computing H_back and H_con')

        self.Ha_back = self.Ha.conjugate()
        self.Hb_back = self.Hb.conjugate()

        self.Ha_con = np.zeros((self.Ha.shape[0:3]) + (15, 15,), dtype=np.complex64)
        self.Hb_con = np.zeros((self.Hb.shape[0:3]) + (15, 15,), dtype=np.complex64)
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(self.compute_H_con)(z) for z in range(self.Ha_con.shape[2])]))

        del self.Ha, self.Hb

    def compute_H_con(self, z):
        self.Ha_con[:, :, z, :, :] = np.einsum('xyjp,xysp->xyjs', self.Ha_back[:, :, z, :, :], self.Ha[:, :, z, :, :])
        self.Hb_con[:, :, z, :, :] = np.einsum('xyjp,xysp->xyjs', self.Hb_back[:, :, z, :, :], self.Hb[:, :, z, :, :])

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

    def recon(self, g, iter_num=10, mod=0):
        log.info('Applying ISRA recon')

        img = (g - g.min()) / (g.max() - g.min())
        imga, imgb = img[..., 0], img[..., 1]

        ek = np.zeros(g.shape[0:3] + (15,))
        ek[..., 0] = 1

        mid_a = self.ConvFFT3(imga, self.Ha_back, order=1)
        mid_b = self.ConvFFT3(imgb, self.Hb_back, order=1)

        if mod == 0:
            for iter in tqdm(range(iter_num)):
                bwd = self.ConvFFT3(ek, self.Ha_con, order=2)
                dif = self.SHDiv(mid_a, bwd)
                del bwd
                ek = self.SHMul(ek, dif)
                del dif

                bwd = self.ConvFFT3(ek, self.Hb_con, order=2)
                dif = self.SHDiv(mid_b, bwd)
                del bwd
                ek = self.SHMul(ek, dif)
                del dif

        if mod == 1:
            for iter in tqdm(range(iter_num)):
                bwd = self.ConvFFT3(ek, self.Ha_con, order=2)
                dif = self.SHDiv(mid_a, bwd)
                del bwd
                ek_a = self.SHMul(ek, dif)
                del dif

                bwd = self.ConvFFT3(ek, self.Hb_con, order=2)
                dif = self.SHDiv(mid_b, bwd)
                del bwd
                ek_b = self.SHMul(ek, dif)
                del dif

                ek = (ek_a + ek_b) / 2

        return ek

    def recon_loss(self, g, iter_num, phant, mod=0):
        g = (g - g.min()) / (g.max() - g.min())
        imga, imgb = g[..., 0], g[..., 1]

        ek = np.zeros(g.shape[0:3] + (15,))
        ek[..., 0] = 1

        mid_a = self.ConvFFT3(imga, self.Ha_back, order=1)
        mid_b = self.ConvFFT3(imgb, self.Hb_back, order=1)

        ssim_rcd = np.zeros(iter_num + 1)
        peak_rcd = np.zeros(iter_num + 1)

        label_f = phant.f
        BinvT, Bvertices = phant.Binv.T, phant.sphere.vertices

        ssim_rcd[0] = eval.SSIM(ek[..., 0], phant.f[..., 0])
        peak_rcd[0] = eval.PeakDif(ek, label_f, BinvT, Bvertices)

        if mod == 0:
            for iter in tqdm(range(iter_num)):
                bwd = self.ConvFFT3(ek, self.Ha_con, order=2)
                dif = self.SHDiv(mid_a, bwd)
                del bwd
                ek = self.SHMul(ek, dif)
                del dif

                bwd = self.ConvFFT3(ek, self.Hb_con, order=2)
                dif = self.SHDiv(mid_b, bwd)
                del bwd
                ek = self.SHMul(ek, dif)
                del dif

                ssim_rcd[iter + 1] = eval.SSIM(ek[..., 0], phant.f[..., 0])
                peak_rcd[iter + 1] = eval.PeakDif(ek, label_f, BinvT, Bvertices)

        if mod == 1:
            for iter in tqdm(range(iter_num)):
                bwd = self.ConvFFT3(ek, self.Ha_con, order=2)
                dif = self.SHDiv(mid_a, bwd)
                del bwd
                ek_a = self.SHMul(ek, dif)
                del dif

                bwd = self.ConvFFT3(ek, self.Hb_con, order=2)
                dif = self.SHDiv(mid_b, bwd)
                del bwd
                ek_b = self.SHMul(ek, dif)
                del dif

                ek = (ek_a + ek_b) / 2

                ssim_rcd[iter + 1] = eval.SSIM(ek[..., 0], phant.f[..., 0])
                peak_rcd[iter + 1] = eval.PeakDif(ek, label_f, BinvT, Bvertices)

        return ssim_rcd, peak_rcd
