import numpy as np
import logging
from tqdm import tqdm
import os

log = logging.getLogger('log')

class Detector:
    """A Detector is specified by its optical axis, numerical aperture, 
    the index of refraction of the sample, and precence of a polarizer.

    By default we use the paraxial approximation.
    """

    def __init__(self, spang, data, optical_axis=[0, 0, 1], lamb=525, na=0.8, n=1.33, FWHM=2000):
        self.spang = spang
        self.data = data
        self.X = spang.X
        self.Y = spang.Y
        self.Z = spang.Z
        self.J = spang.J
        self.P = data.P
        self.V = data.V
        self.lamb = lamb

        self.optical_axis = optical_axis
        self.na = na
        self.n = n
        self.ls_sigma = FWHM / 2.3548

        self.Gaunt_633 = np.load(os.path.join(os.path.dirname(__file__),
                                              '../harmonics/gaunt_633.npy'))
        self.Gaunt = np.load(os.path.join(os.path.dirname(__file__), '../harmonics/gaunt_l4.npy'))

        self.rotate = np.array([[1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, -1 / 2, 0, np.sqrt(3) / 2],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, np.sqrt(3) / 2, 0, 1 / 2]])

    def calc_H(self):
        mtx = np.zeros((self.X, self.Y, self.Z, 6), dtype=np.complex64)

        if self.optical_axis == [0, 0, 1]:  # z-detection
            rz = np.fft.rfftfreq(self.Z, 1 / self.Z) * self.data.vox_dim[2]
            hz = np.exp(-(rz ** 2) / (2 * self.ls_sigma ** 2), dtype=np.float32)

            temp = np.zeros((self.X, self.Y, rz.shape[0], 6), dtype=np.complex64)
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1, backend='threading')(tqdm(
                [delayed(self.compute_sh_det0)(temp, z, r, hz) for z, r in enumerate(rz)]))

            start = slice(0, (self.Z // 2) + 1)
            end = slice(None, -(self.Z // 2), -1)

            mtx[:, :, start, :] = temp
            mtx[:, :, end, :] = temp[:, :, 1:-1, :]
            mtx = mtx * 4 * np.pi / 3

        if self.optical_axis == [1, 0, 0]:  # x-detection
            rx = np.fft.rfftfreq(self.X, 1 / self.X) * self.data.vox_dim[0]
            hx = np.exp(-(rx ** 2) / (2 * self.ls_sigma ** 2), dtype=np.float32)

            temp = np.zeros((rx.shape[0], self.Y, self.Z, 6), dtype=np.complex64)
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1, backend='threading')(
                tqdm([delayed(self.compute_sh_det1)(temp, x, r, hx) for x, r in enumerate(rx)]))

            start = slice(0, (self.X // 2) + 1)
            end = slice(None, -(self.X // 2), -1)
            mtx[start, :, :, :] = temp
            mtx[end, :, :, :] = temp[1:-1, :, :, :]

            mtx = mtx * 4 * np.pi / 3
            mtx = np.einsum('rs,xyzs->xyzr', self.rotate, mtx)

        mtx = np.fft.rfftn(np.real(mtx), axes=(0, 1, 2))
        return mtx

    def compute_sh_det0(self, mtx, z, r, hz):
        e = [1, -1, 0]
        for j in range(3):
            for j_ in range(3):
                Glm = self.Gaunt_633[:, e[j] + 1, e[j_] + 1]
                mtx[:, :, z, :] = mtx[:, :, z, :] + np.einsum('s,xy->xys', Glm, self.cal_B_mtx(j, j_, r))
        mtx[:, :, z, :] = mtx[:, :, z, :] * hz[z]

    def compute_sh_det1(self, mtx, x, r, hx):
        e = [1, -1, 0]
        for j in range(3):
            for j_ in range(3):
                Glm = self.Gaunt_633[:, e[j] + 1, e[j_] + 1]
                mtx[x, :, :, :] = mtx[x, :, :, :] + np.einsum('s,yz->yzs', Glm, self.cal_B_mtx(j, j_, r))
        mtx[x, :, :, :] = mtx[x, :, :, :] * hx[x]

    def cal_B_mtx(self, j, j_, r):
        mtx = self.cal_beta_mtx(0, j, r) * self.cal_beta_mtx(0, j_, r).conjugate() + \
              self.cal_beta_mtx(1, j, r) * self.cal_beta_mtx(1, j_, r).conjugate()
        return mtx

    def cal_beta_mtx(self, i, j, r):
        g = self.list_g()[i][j]
        if self.optical_axis == [0, 0, 1]:  # z-detection
            vm = self.n / self.lamb
            vc = self.na * 2 / self.lamb
            dx = np.fft.rfftfreq(self.X, d=self.data.vox_dim[0])
            dy = np.fft.rfftfreq(self.Y, d=self.data.vox_dim[1])

            mx, my = np.meshgrid(dx, dy)
            mx, my = mx.T, my.T
            nu = np.sqrt(mx ** 2 + my ** 2)
            nu[nu >= vc / 2] = vm - np.finfo(np.float32).eps
            nu_phi = np.arctan2(my, mx)
            A_mat = self.A(nu, vm)
            A_mat[nu >= vc / 2] = 0
            g_mat = g(nu, nu_phi, vm)
            Phi_mat = self.Phi(nu, r, vm)

            temp = A_mat * g_mat * Phi_mat
            xstart = slice(0, (self.X // 2) + 1)
            xend = slice(None, -(self.X // 2), -1)
            ystart = slice(0, (self.Y // 2) + 1)
            yend = slice(None, -(self.Y // 2), -1)
            mtx = np.zeros((self.X, self.Y), dtype=temp.dtype)
            mtx[xstart, ystart] = temp
            mtx[xend, ystart] = temp[1:-1, :]
            mtx[xstart, yend] = temp[:, 1:-1]
            mtx[xend, yend] = temp[1:-1, 1:-1]

            return np.fft.ifftn(mtx, axes=(0, 1))

        if self.optical_axis == [1, 0, 0]:  # x-detection
            vm = self.n / self.lamb
            vc = self.na * 2 / self.lamb

            dy = np.fft.rfftfreq(self.Y, d=self.data.vox_dim[1])
            dz = np.fft.rfftfreq(self.Z, d=self.data.vox_dim[2])

            my, mz = np.meshgrid(dy, dz)
            my, mz = my.T, mz.T
            nu = np.sqrt(my ** 2 + mz ** 2)
            nu[nu >= vc / 2] = vm - np.finfo(np.float32).eps
            nu_phi = np.arctan2(mz, my)
            A_mat = self.A(nu, vm)
            A_mat[nu >= vc / 2] = 0
            g_mat = g(nu, nu_phi, vm)
            Phi_mat = self.Phi(nu, r, vm)

            temp = A_mat * g_mat * Phi_mat
            ystart = slice(0, (self.Y // 2) + 1)
            yend = slice(None, -(self.Y // 2), -1)
            zstart = slice(0, (self.Z // 2) + 1)
            zend = slice(None, -(self.Z // 2), -1)
            mtx = np.zeros((self.Y, self.Z), dtype=temp.dtype)
            mtx[ystart, zstart] = temp
            mtx[yend, zstart] = temp[1:-1, :]
            mtx[ystart, zend] = temp[:, 1:-1]
            mtx[yend, zend] = temp[1:-1, 1:-1]

            return np.fft.ifftn(mtx, axes=(0, 1))

    def list_g(self):
        def g00(nu, nu_phi, vm):
            return np.sin(nu_phi) ** 2 + np.cos(nu_phi) ** 2 * np.sqrt(1 - (nu / vm) ** 2)

        def g10(nu, nu_phi, vm):
            return 0.5 * np.sin(nu_phi * 2) * (np.sqrt(1 - (nu / vm) ** 2) - 1)

        def g01(nu, nu_phi, vm):
            return 0.5 * np.sin(nu_phi * 2) * (np.sqrt(1 - (nu / vm) ** 2) - 1)

        def g11(nu, nu_phi, vm):
            return np.cos(nu_phi) ** 2 + np.sin(nu_phi) ** 2 * np.sqrt(1 - (nu / vm) ** 2)

        def g02(nu, nu_phi, vm):
            return nu * np.cos(nu_phi)

        def g12(nu, nu_phi, vm):
            return nu * np.sin(nu_phi)

        return [[g00, g01, g02], [g10, g11, g12]]

    def A(self, nu, vm):
        return (1 - (nu / vm) ** 2) ** (-0.25)

    def Phi(self, nu, rop, vm):
        return np.exp(1.0j * 2 * np.pi * rop * np.sqrt(vm ** 2 - nu ** 2))