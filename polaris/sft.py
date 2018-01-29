from polharmonic.util import *
import numpy as np
from sympy import *
import sympy.functions.special.spherical_harmonics as sh

# Globals
theta = Symbol('theta', real=True)
phi = Symbol('phi', real=True)

# Analytic forward spherical Fourier transform
def sft(f, max_l=4):
    coeffs = []
    for l in range(0, max_l+2, 2):
        for m in range(-l, l+1):
            print("Integrating: "+ str(l) + ', ' + str(m))
            Znm = sh.Znm(l, m, theta, phi).expand(func=True)
            theta_int = integrate(expand(sin(theta)*Znm*f), (theta, 0, pi)) 
            final_int = integrate(expand_trig(theta_int.rewrite(cos)), (phi, 0, 2*pi))
            coeffs.append(re(final_int).evalf())
    return np.array(coeffs)

# Numerical forward spherical Fourier transform from delta
def tp_sft(tp, max_l=4):
    labels = []
    coeffs = []
    for l in range(0, max_l+2, 2):
        for m in range(-l, l+1):
            coeffs.append(spZnm(l, m, tp[0], tp[1]))
    return np.array(coeffs)

def field_tp_sft(tp_field, max_l=4):
    return np.apply_along_axis(tp_sft, 2, tp_field, max_l=max_l)

# Analytic inverse spherical Fourier transform
def isft(ylm):
    f = 0
    for y in ylm:
        f += y['expr']*sh.Znm(y['l'], y['m'], theta, phi).expand(func=True)
    return f

# Tests
def test_transforms():
    h1 = 1 - (sin(Theta)*cos(Phi)*sin(theta)*cos(phi) + sin(Theta)*sin(Phi)*sin(theta)*sin(phi) + cos(Theta)*cos(theta))**2
    h2 = (sin(theta)**2)*(cos(phi - phi_pol)**2)*2*(A + B*(1 - (sin(theta)**2)*(cos(phi)**2)))
    h3 = ((sin(theta)*sin(phi)*sin(phi_pol) - cos(theta)*cos(phi_pol))**2)*2*(A + B*(sin(theta)**2))
    h4 = sin(theta)**2 + 2
    h5 = sin(theta)**4*(cos(phi)**2)

    for h in [h1, h2, h3, h4, h5]:
        Hlm = sft(h, max_l=4) # Increase max_l if result it not zero
        h2 = isft(Hlm)
        print(simplify(h - h2.rewrite(cos))) # Should be zero
