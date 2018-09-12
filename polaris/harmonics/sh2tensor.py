# Precompute the matrix that calculates the rank-2 tensor entries from spherical
# harmonic coeffients.
#
# Tensor entry order [Dxx, Dyy, Dzz, Dxy, Dyz, Dxz]

from polaris import util as myutil
import numpy as np
from sympy import *
import sympy.functions.special.spherical_harmonics as sh

def calc_sh2tensor(filename, lmax=2):
    jmax = myutil.maxl2maxj(lmax)
    H = np.zeros((6, jmax))
    theta = Symbol('theta', real=True)
    phi = Symbol('phi', real=True)
    elems = [(cos(phi)*sin(theta))**2, (sin(phi)*sin(theta))**2, cos(theta)**2,
               cos(phi)*sin(phi)*(sin(theta)**2), sin(phi)*sin(theta)*cos(theta),
               cos(phi)*sin(theta)*cos(theta)]
    for i, elem in enumerate(elems):
        for j in range(jmax):
            l, m = myutil.j2lm(j)
            Znm = sh.Znm(l, m, theta, phi).expand(func=True)
            theta_int = integrate(sin(theta)*Znm*elem, (theta, 0 , pi/2))
            final_int = integrate(expand_trig(theta_int.rewrite(cos)), (phi, 0, 2*pi))
            H[i,j] = final_int.evalf()
            print(i, l, m, final_int)
    import pdb; pdb.set_trace() 
    np.save(filename, H)

calc_sh2tensor('sh2tensor.npy', lmax=6)
