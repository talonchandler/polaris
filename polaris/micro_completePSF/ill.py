import numpy as np
import polaris.harmonics.shcoeffs as sh
import polaris.util as util
import logging
import os

log = logging.getLogger('log')


class Illuminator:
    """An Illuminator is specified by its optical axis, numerical aperture, 
    the index of refraction of the sample, and polarizer orientation.

    By default we use the paraxial approximation.
    """

    def __init__(self, data, optical_axis=[1, 0, 0]):
        self.data = data
        self.P = data.P
        self.optical_axis = optical_axis

    def calc_H(self):
        sh_ills = []
        if self.optical_axis == [1, 0, 0]:
            sh_ills = np.zeros((self.P, 6))
            for p in range(self.P):
                pol = self.data.pols_norm[0, p, :]
                sh_ills[p, :] = self.H(pol).coeffs

        if self.optical_axis == [0, 0, 1]:
            sh_ills = np.zeros((self.P, 6))
            for p in range(self.P):
                pol = self.data.pols_norm[1, p, :]
                sh_ills[p, :] = self.H(pol).coeffs

        return sh_ills

    def H(self, pol=None):
        if pol is None:  # For normalization
            cc = sh.SHCoeffs([1, 0, 0, -1 / np.sqrt(5), 0, 0]) / np.sqrt(4 * np.pi)
            if self.optical_axis == [1, 0, 0]:  # x-illumination
                cc = cc.rotate()
            return cc
        out = []
        for j in range(6):
            l, m = util.j2lm(j)
            theta, phi = util.xyz2tp(*pol)
            if l == 0:
                cc = 1.0
            else:
                cc = 0.4
            out.append(cc * util.spZnm(l, m, theta, phi))
        return sh.SHCoeffs(out)
