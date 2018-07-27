import numpy as np
import subprocess
from polaris import util
from polaris.harmonics import gaunt
import matplotlib.pyplot as plt
from PIL import Image
import os
import itertools

# Compute coefficients
# gaunt.calc_chtriple_tensor('chcoeff_n2.npy', nmax=2) # Expensive precomputation
P = np.load(os.path.join(os.path.dirname(__file__), 'chcoeff_n2.npy')) 

class CHCoeffs:
    """A CHCoeffs object stores real circular harmonic coefficients for even bands.
    It provides methods for adding, multiplying, and plotting these
    coefficients.

    Uses the following "lexicographic ordering" of the even circular harmonics:

    z_0, z_-2, z_2, z_-4, z_4, ...

    """

    def __init__(self, coeffs):
        self.nmax = util.i2n(len(coeffs) - 1)
        self.coeffs = coeffs

    def __add__(self, other):
        return CHCoeffs(self.coeffs + other.coeffs)
        
    def __mul__(self, other):
        if not isinstance(other, CHCoeffs):
            return SHCoeffs(np.array(self.coeffs)*other)

        # Pad inputs
        x1 = np.pad(np.array(self.coeffs), (0, 2), 'constant')
        x2 = np.pad(np.array(other.coeffs), (0, 2), 'constant')

        # Multiply
        result = np.dot(np.dot(P, x1), x2)
        return CHCoeffs(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, scalar):
        return CHCoeffs(self.coeffs/scalar)

    def __repr__(self):
        string = 'CHCoeffs: \n'
        string += str(self.coeffs) + '\n'
        return string
    
    # def plot(self, folder=''):
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
        
    #     self.plot_dist(filename=folder+'/dist.png')
    #     self.plot_spectrum(filename=folder+'/spectrum.pdf')

    # def plot_spectrum(self, filename='spectrum.pdf'):
    #     print('Plotting: ' + filename)
    #     f, ax = plt.subplots(1, 1, figsize=(4, 4))

    #     # Create image of spherical harmonic coefficients
    #     image = np.zeros((self.rmax, self.mmax))
    #     for j, c in enumerate(self.coeffs):
    #         l, m = util.j2lm(j)
    #         image[int(l/2), self.lmax + m] = c

    #     # Label rows and columns
    #     for l in range(self.lmax + 1):
    #         if l == 0:
    #             prepend = 'l='
    #         else:
    #             prepend = ''
    #         if l%2 == 0:
    #             ax.annotate(r'$'+prepend+str(l)+'$', xy=(1, 1), xytext=(-0.75, int(l/2)),
    #                          textcoords='data', ha='right', va='center')
                
    #     ax.annotate(r'$m=$', xy=(1, 1), xytext=(-0.75, -0.75),
    #                  textcoords='data', ha='right', va='center')
    #     for m in range(2*self.lmax + 1):
    #         ax.annotate('$'+str(m - self.lmax)+'$', xy=(1, 1),
    #                      xytext=(int(m), -0.75),
    #                      textcoords='data', ha='center', va='center')

    #     # Label each pixel
    #     for (y,x), value in np.ndenumerate(image):
    #         if value != 0:
    #             ax.annotate("{0:.2f}".format(value), xy=(1, 1), xytext=(x, y),
    #                      textcoords='data', ha='center', va='center')
            
    #     ax.imshow(image, cmap='bwr', interpolation='nearest',
    #               vmin=-np.max(self.coeffs), vmax=np.max(self.coeffs))
    #     ax.axis('off')

    #     f.savefig(filename, bbox_inches='tight')

    # def plot_dist(self, filename='dist.png', n_pts=2500, r=1, mag=1, show=False):
    #     from mayavi import mlab
    #     print('Plotting: ' + filename)
        
    #     # Calculate radii
    #     tp = util.fibonacci_sphere(n_pts)
    #     xyz = util.fibonacci_sphere(n_pts, xyz=True)
    #     radii = np.zeros(tp.shape[0])
    #     for i, c in enumerate(self.coeffs):
    #         l, m = util.j2lm(i)
    #         radii += c*util.spZnm(l, m, tp[:,0], tp[:,1])
    #     radii = radii/np.max(radii)
        
    #     # Split into positive and negatives
    #     n = radii.clip(max=0) 
    #     p = radii.clip(min=0)*(-1)

    #     # Triangulation
    #     from scipy.spatial import ConvexHull
    #     ch = ConvexHull(xyz)
    #     triangles = ch.simplices

    #     # Create figure
    #     mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
    #     mlab.clf()
        
    #     # Plot
    #     mlab.triangular_mesh(p*xyz[:,0], p*xyz[:,1], p*xyz[:,2], triangles, color=(1, 0, 0))
    #     s = mlab.triangular_mesh(n*xyz[:,0], n*xyz[:,1], n*xyz[:,2], triangles, color=(0, 0, 1))
    #     s.scene.light_manager.light_mode = "vtk"
        
    #     # View and save
    #     mlab.view(azimuth=45, elevation=45, distance=5, focalpoint=None,
    #               roll=None, reset_roll=True, figure=None)
    #     mlab.savefig(filename, magnification=mag)
    #     subprocess.call(['convert', filename, '-transparent', 'white', filename])
    #     if show:
    #         mlab.show()
