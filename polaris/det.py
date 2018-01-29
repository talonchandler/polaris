import numpy as np
import polharmonic.util as util
from sympy import *

class Detector:
    """An Detector is specified by its numerical aperture, optical axis, and the
    index of refraction of the sample.
    """
    def __init__(self, theta_optical_axis=0, NA=0.8, n=1.33):
        self.NA = NA
        self.theta_optical_axis = theta_optical_axis        
        self.n = n

    def det_prf(self):
        theta = Symbol('theta', real=True)
        phi = Symbol('phi', real=True)
        A, B = NAn2AB(self.NA, self.n)
        if self.theta_optical_axis == 0:
            return 2*(A + B*(sin(theta)**2))
        elif self.theta_optical_axis == 90:
            return 2*(A + B*(1 - (sin(theta)**2)*(cos(phi)**2)))
        else:
            print('Warning: theta_optical_axis must be 0 or 90')
            return 0
        

# Detector helper functions
def NAn2AB(NA, n):
    alpha = np.arcsin(NA/n)
    return alpha2A(alpha), alpha2B(alpha)

def alpha2A(alpha):
    ca = np.cos(alpha)
    return 0.25 - 0.375*ca + 0.125*(ca**3)

def alpha2B(alpha):
    ca = np.cos(alpha)
    return 0.1875*ca + 0.1875*(ca**3)
