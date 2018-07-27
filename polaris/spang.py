from polaris import viz, util
import numpy as np
from dipy.viz import window, actor

class Spang:
    """
    A Spang (short for spatio-angular density) is a representation of a 
    spatio-angular density f(r, s) stored as a 4D array of voxel values 
    and spherical harmonic coefficients [x, y, z, j]. A Spang object is 
    a discretized member of object space U. 
    """
    def __init__(self, f=np.zeros((3,3,3,1)), vox_dim=(1,1,1)):
        self.NX = f.shape[0]
        self.NY = f.shape[1]
        self.NZ = f.shape[2]
        
        # Calculate band dimensions
        self.lmax, mm = util.j2lm(f.shape[-1] - 1)
        self.S = util.maxl2maxj(self.lmax)

        # Fill the rest of the last l band with zeros
        if f.shape[-1] != self.S:
            temp = np.zeros((f.shape[0], f.shape[1], f.shape[2], self.S))
            temp[:,:,:,:f.shape[-1]] = f
            self.f = temp
        else:
            self.f = f

        self.vox_dim = vox_dim

    def save_mips(self, filename='spang_mips.pdf'):
        col_labels = np.apply_along_axis(util.j2str, 1, np.arange(self.S)[:,None])[None,:]
        viz.plot5d(filename, self.f[:,:,:,:,None], col_labels=col_labels)

    def save_fft(self, filename='fft.pdf'):
        axes = (0,1,2)
        myfft = np.abs(np.fft.fftn(self.f, axes=axes, norm='ortho'))
        ffts = np.fft.fftshift(myfft, axes=axes)
        col_labels = np.apply_along_axis(util.j2str, 1, np.arange(self.S)[:,None])[None,:]
        viz.plot5d(filename, ffts[:,:,:,:,None], col_labels=col_labels)
        
    def visualize(self, out_path='out/', 
                  scale=0.5, n_frames=1, size=(600, 600), mag=4, interact=False):

        # Calculate odf from sh coefficients
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')
        B = np.zeros((len(sphere.theta), self.f.shape[-1]))
        for index, x in np.ndenumerate(B):
            l, m = util.j2lm(index[1])
            B[index] = util.spZnm(l, m, sphere.theta[index[0]], sphere.phi[index[0]])
        odf = np.einsum('ijkl,ml->ijkm', self.f, B)

        # Render
        ren = window.Renderer()
        ren.background([1,1,1])
        fodf_spheres = viz.odf_slicer(odf, sphere=sphere, scale=scale,
                                      norm=False, colormap='plasma', mask=None)
        ren.add(fodf_spheres)

        import os
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        window.record(ren, out_path=out_path, size=size,
                      az_ang=1, n_frames=n_frames, path_numbering=True,
                      magnification=mag, verbose=True)

        if interact:
            window.show(ren)
