import numpy as np
import polharmonic.util as util
from sympy import *

class Illuminator:
    """An Illuminator is specified by its optical axis and polarization orientation.
    """
    def __init__(self, theta_optical_axis=0, phi_pol=0):
        self.theta_optical_axis = theta_optical_axis
        self.phi_pol = phi_pol

    def exc_prf(self):
        theta = Symbol('theta', real=True)
        phi = Symbol('phi', real=True)
        if self.theta_optical_axis == 0:
            return (sin(theta)**2)*(cos(phi - np.deg2rad(self.phi_pol))**2)
        elif self.theta_optical_axis == 90:
            return ((sin(theta)*sin(phi)*sin(np.deg2rad(self.phi_pol)) - cos(theta)*cos(np.deg2rad(self.phi_pol)))**2)
        else:
            print('Warning: theta_optical_axis must be 0 or 90')
            return 0
