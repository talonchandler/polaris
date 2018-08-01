import numpy as np
from scipy.special import sph_harm, hyp1f1

# SciPy real spherical harmonics with identical interface to SymPy's Znm
# Useful for fast numerical evaluation of Znm
def spZnm(l, m, theta, phi):
    if m > 0:
        return np.real((sph_harm(m, l, phi, theta) +
                np.conj(sph_harm(m, l, phi, theta)))/(np.sqrt(2)))
    elif m == 0:
        return np.real(sph_harm(m, l, phi, theta))
    elif m < 0:
        return  -np.real((sph_harm(m, l, phi, theta) -
                 np.conj(sph_harm(m, l, phi, theta)))/(np.sqrt(2)*1j))

# Calculate spherical harmonic coefficients of delta
def xyz_sft(xyz, max_l=4):
    if xyz[0] == 0 and xyz[1] == 0 and xyz[2] == 0:
        return np.zeros(maxl2maxj(max_l))
    tp = xyz2tp(xyz[0], xyz[1], xyz[2])
    coeffs = []
    for l in range(0, max_l+2, 2):
        for m in range(-l, l+1):
            coeffs.append(spZnm(l, m, tp[0], tp[1]))
    return np.array(coeffs)

# Calculate spherical harmonic coefficients of delta with kappa (Watson dist)
def xyzk_sft(xyzk, max_l=4, B=None):
    xyz = xyzk[0:-1]
    xyz = xyz/np.linalg.norm(xyz)
    k = xyzk[-1]
    if k > 10 or k == 0:
        return xyz_sft(xyzk[0:-1], max_l=max_l)
    else:
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')
        xx = sphere.x
        yy = sphere.y
        zz = sphere.z
        dirs = np.stack([xx, yy, zz], axis=-1)

        # Evaluate Watson distribution
        watson = np.exp(k*np.dot(dirs, xyz)**2)/(4*np.pi*hyp1f1(0.5, 1.5, k))

        # Return shcoeffs
        return np.dot(B.T, watson)

# Convert between spherical harmonic indices (l, m) and multi-index (j)
def j2lm(j):
    if j < 0:
        return None
    l = 0
    while True:
        x = 0.5*l*(l+1)
        if abs(j - x) <= l:
            return l, int(j-x)
        else:
            l = l+2

def lm2j(l, m):
    if abs(m) > l or l%2 == 1:
        return None
    else:
        return int(0.5*l*(l+1) + m)

def maxl2maxj(l):
    return int(0.5*(l + 1)*(l + 2))

# Convert between Cartesian and spherical coordinates
def tp2xyz(tp):
    theta = tp[0]
    phi = tp[1]
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)

def xyz2tp(x, y, z):
    arccos_arg = z/np.sqrt(x**2 + y**2 + z**2)
    if np.isclose(arccos_arg, 1.0): # Avoid arccos floating point issues
        arccos_arg = 1.0
    elif np.isclose(arccos_arg, -1.0):
        arccos_arg = -1.0
    return np.arccos(arccos_arg), np.arctan2(y, x)

# Convert xyz to string for plots
def xyz2str(xyz):
    string = '$'
    s = ['\hat{\mathbf{x}}', '\hat{\mathbf{y}}', '\hat{\mathbf{z}}']
    for i, element in enumerate(xyz):
        if element == 1:
            string += '+'+s[i]
        if element == -1:
            string += '-'+s[i]
    if string[1] == '+' or string == '$-\hat{\mathbf{z}}':
        string = string[0] + string[2:]
    string += '$'
    return np.array(string, dtype=object)

# Convert j to string for plots
def j2str(j):
    string = '$y_{'
    l, m = j2lm(j)
    string += str(l) + '}^{' + str(m) + '}$'
    return np.array(string, dtype=object)

# Absolute max projection
def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

# Returns "equally" spaced points on a unit sphere in spherical coordinates.
# http://stackoverflow.com/a/26127012/5854689
def fibonacci_sphere(n, xyz=False):
    z = np.linspace(1 - 1/n, -1 + 1/n, num=n) 
    theta = np.arccos(z)
    phi = np.mod((np.pi*(3.0 - np.sqrt(5.0)))*np.arange(n), 2*np.pi) - np.pi
    if xyz:
        return np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T
    else:
        return np.vstack((theta, phi)).T

# Kullback-Leibler distance (see Barrett 15.151)
def kl(g, g0):
    return np.sum(g0 - g + g*np.log(g/g0))
