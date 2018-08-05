from polaris import util, viz, data, spang
from polaris.micro import ill, det, micro
from polaris.harmonics import shcoeffs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import os
import subprocess
from tqdm import tqdm

class MultiMicroscope:
    """A MultiMicroscope represents an experiment that collects intensity data 
    under several different conditions (different polarization states or 
    illumination schemes).

    A MultiMicroscope mainly consists of a list of Microscopes.
    """
    def __init__(self, spang, data, ill_optical_axes=[[1,0,0], [0,0,1]],
                 det_optical_axes=[[0,0,1], [1,0,0]],
                 ill_nas=2*[0], det_nas=2*[0.8], n_samp=1.33, sigma_ax=0.33,
                 lamb=525, spang_coupling=True):

        self.spang = spang
        self.data = data
        m = [] # List of microscopes

        # Cycle through paths
        for i, det_optical_axis in enumerate(det_optical_axes):
            ill_ = ill.Illuminator(optical_axis=ill_optical_axes[i],
                                   na=ill_nas[i], n=n_samp)
            det_ = det.Detector(optical_axis=det_optical_axes[i],
                                na=det_nas[i], n=n_samp, sigma_ax=sigma_ax)
            m.append(micro.Microscope(ill=ill_, det=det_, spang_coupling=spang_coupling)) # Add microscope

        self.micros = m
        self.N = len(m)
        self.jmax = m[0].h(0, 0, 0).jmax
        self.nbranch = self.N*self.micros[0].n_len
        self.lamb = lamb

    def calc_point_H(self, vx, vy, vz, v, pols):
        tf = self.micros[v].H(vx,vy,vz).coeffs
        # Take cft
        if v == 0:
            x = pols[v,:,2]
            y = pols[v,:,1]
        if v == 1:
            x = pols[v,:,0]
            y = pols[v,:,1]
        # TODO: Move this to chcoeffs
        out = np.outer(tf[0,:], np.ones(4)) + np.outer(tf[1,:], 2*x*y) + np.outer(tf[2,:], x**2 - y**2)
        # if tf[0,0] != 0 and v == 1:
        #     import pdb; pdb.set_trace()
        return out # return s x p

    def calc_H(self):
        H = np.zeros((self.spang.NX, self.spang.NY, self.spang.NZ, self.jmax,
                      self.data.P, self.data.V), dtype=np.float32)

        print('Computing H')
        for v, view in enumerate(self.data.views):
            dx = np.fft.fftshift(np.fft.fftfreq(self.spang.NX, d=self.data.vox_dim[0]))*self.lamb/self.micros[v].det.na
            dy = np.fft.fftshift(np.fft.fftfreq(self.spang.NY, d=self.data.vox_dim[1]))*self.lamb/self.micros[v].det.na
            dz = np.fft.fftshift(np.fft.fftfreq(self.spang.NZ, d=self.data.vox_dim[2]))*self.lamb/self.micros[v].det.na
            for x, nux in enumerate(tqdm(dx)):
                for y, nuy in enumerate(tqdm(dy)):
                    for z, nuz in enumerate(dz):
                        H[x,y,z,:,:,v] = self.calc_point_H(nux, nuy, nuz, v, self.data.pols_norm)

        self.H = H/np.max(np.abs(H))

        print('Computing SVD')
        oldshape = self.H.shape
        newshape = self.H.shape[0:4] + (self.H.shape[4]*self.H.shape[5],)
        HH = np.reshape(self.H, newshape)
        u, s, vh = np.linalg.svd(HH, full_matrices=False)

        # Check svd
        # HH = np.reshape(u @ (s[...,None] * vh), self.H.shape)
        mask = np.sum(s, axis=(0,1,2)) > 1e-10  # remove small singular values
        self.u = u[...,mask]
        self.s = s[...,mask]
        self.vh = vh[...,mask,:]

    def save_H(self, filename):
        np.savez(filename, H=self.H, u=self.u, s=self.s, vh=self.vh)
        
    def load_H(self, filename):
        files = np.load(filename)
        self.H = files['H']
        self.u = files['u']
        self.s = files['s']
        self.vh = files['vh']

    def plot_H(self, filename='H.pdf', s=0):
        row_labels = 'View = ' + np.apply_along_axis(util.xyz2str, 1, self.data.views)
        col_labels = 'Polarizer = ' + np.apply_along_axis(util.xyz2str, 2, self.data.pols)
        yscale_label = str(1/self.data.yscale) + ' $\mu$m${}^{-1}$'
        viz.plot5d(filename, self.H[:,:,:,s,:,:], row_labels, col_labels, yscale_label, force_bwr=True)

    def fwd(self, f, fast=True, snr=None):
        # Truncate spang for angular bandlimit
        f = f[:,:,:,:self.jmax]
        
        # 3D FT
        F = np.fft.fftn(f, axes=(0,1,2))
        Fshift = np.fft.fftshift(F, axes=(0,1,2))
        
        # Tensor multiplication
        if fast:
            outshape = self.H.shape[0:3] + self.H.shape[4:]
            Gshift = np.reshape(np.einsum('ijklm,ijkl,ijknl,ijkn->ijkm',
                                          self.vh, self.s, self.u, Fshift), outshape)
        else:
            Gshift = np.einsum('ijklmn,ijkl->ijkmn', self.H, Fshift)
        
        # 3D IFT
        G = np.fft.ifftshift(Gshift, axes=(0,1,2))
        g = np.real(np.fft.ifftn(G, axes=(0,1,2)))

        # Apply Poisson noise
        if snr is not None:
            norm = snr**2/np.max(g)
            arr_poisson = np.vectorize(np.random.poisson)
            g = arr_poisson(g*norm)/norm

        # return g
        return g/np.max(g)

    def adj(self, g):
        # 3D FT
        G = np.fft.fftn(g, axes=(0,1,2))
        Gshift = np.fft.fftshift(G, axes=(0,1,2))
        
        # Tensor multiplication
        Fshift = np.einsum('ijklmn,ijkmn->ijkl', self.H, Gshift)
        
        # 3D IFT        
        F = np.fft.ifftshift(Fshift, axes=(0,1,2))
        f = np.real(np.fft.ifftn(F, axes=(0,1,2)))

        return f

    def pinv(self, g, eta=0):
        # Regularize and truncate singular values
        sreg = np.where(self.s > 1e-10, self.s/(self.s**2 + eta), 0) 
        
        # 3D FT
        G = np.fft.fftn(g, axes=(0,1,2))
        Gshift = np.fft.fftshift(G, axes=(0,1,2))
        
        # Tensor multiplication
        Gshift2 = np.reshape(Gshift, G.shape[0:3] + (G.shape[3]*G.shape[4],))
        Fshift = np.einsum('ijklm,ijkm,ijkmn,ijkn->ijkl',
                           self.u, sreg, self.vh, Gshift2)
        
        # 3D IFT        
        F = np.fft.ifftshift(Fshift, axes=(0,1,2))
        f = np.real(np.fft.ifftn(F, axes=(0,1,2)))

        return f

    def pmeas(self, f, eta=0):
        return self.pinv(self.fwd(f), eta=eta)

    def pnull(self, f):
        return f - self.pmeas(f) # could be made more efficient
    
    def calc_SVD(self, n_px=2**6):
        w = 2.0
        self.xcoords = np.linspace(-w, w, n_px),
        X, Y, Z = np.mgrid[-w:w:n_px*1j, -w:w:n_px*1j, -w:w:n_px*1j]
        
        # For each position calculate K and solve eigenequation
        sigma = np.zeros((n_px, n_px, n_px, self.nbranch))
        for index, x in np.ndenumerate(X):
            u, s, v = self.calc_point_SVD(x, Y[index], Z[index])
            sigma[index[0], index[1], index[2], :] = s

        self.sigma = sigma
        self.sigma_max = np.max(sigma)

    def calc_point_SVD(self, x, y, z):
        # TODO: Generalize to n views
        HH0 = self.micros[0].H(x, y, z).coeffs
        HH1 = self.micros[1].H(x, y, z).coeffs
        HH = np.concatenate((HH0, HH1))
            
        K = np.dot(HH, HH.T)
        mu, v = np.linalg.eigh(K)
        u = np.dot(HH.T, v)

        return u[:,::-1], np.sqrt(mu[::-1]), v[:,::-1] # Transpose

    def plot_SVS_3D(self, filename='svs.pdf', n_px=2**6, marks=np.array([[0,0,0], [0.5,0,0], [1,0,0], [1.5,0,0]])):
        print('Plotting: ' + filename)

        # Layout windows
        inches = 1.5
        rows = self.sigma.shape[-1] + 1
        cols = marks.shape[0] + 1
        f = plt.figure(figsize=(inches*cols, inches*(rows - 0.75)))
        widths = cols*[1]
        heights = [1]*(rows - 1) + [0.05]
        spec = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths,
                                 height_ratios=heights)

        # For saving singular function pngs
        folder = 'singular'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # For each branch
        # self.sigma[0,0,0,:] = 10 # Marker for testing
        # self.sigma[-1,0,0,:] = 10 # Marker for testing
        for row in range(rows):
            for col in range(cols):
                if col == 0 and row != rows - 1:
                    mini_spec = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=spec[row, col])
                    for a in range(2):
                        for b in range(2):
                            # Annotate 
                            ax = f.add_subplot(mini_spec[a, b])
                            ax.axis('off')
                            sigma = self.sigma[:,:,:,row]/self.sigma_max
                            sigma_plot = np.zeros(sigma.shape[0:2])

                            if a == 0 and b == 1:
                                ax.annotate('', xy=(-0.1,-0.1), xytext=(-0.6, -0.1), xycoords='axes fraction', textcoords='axes fraction', va='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
                                ax.annotate('', xy=(-0.1,-0.1), xytext=(-0.1, 0.4), xycoords='axes fraction', textcoords='axes fraction', ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
                                ax.annotate('', xy=(-0.1,-0.1), xytext=(-0.1, -0.6), xycoords='axes fraction', textcoords='axes fraction', ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
                                ax.annotate('', xy=(-0.1,-0.1), xytext=(0.4, -0.1), xycoords='axes fraction', textcoords='axes fraction', va='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
                                ax.annotate('$z$', xy=(0,0), xytext=(-0.7, -0.1), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=8)
                                ax.annotate('$y$', xy=(0,0), xytext=(-0.1, 0.5), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=8)
                                ax.annotate('$z$', xy=(0,0), xytext=(-0.1, -0.7), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=8)
                                ax.annotate('$x$', xy=(0,0), xytext=(0.5, -0.1), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=8)
                                sigma_plot = np.max(sigma, axis=2).T
                                ax.plot(marks[:,0], marks[:,1], 'xk', ms=1.5, mew=0.2)
                            if a == 1 and b == 1:
                                sigma_plot = np.max(sigma, axis=1)[:,::-1].T
                                ax.plot(marks[:,0], marks[:,2], 'xk', ms=1.5, mew=0.2)
                            if a == 0 and b == 0:
                                sigma_plot = np.max(sigma, axis=0)[:,::-1]
                                ax.plot(marks[:,2], marks[:,1], 'xk', ms=1.5, mew=0.2)                                
                            ax.imshow(sigma_plot, cmap="bwr", vmin=-1, vmax=1, interpolation='none', extent=[-2,2,-2,2], origin='lower')
                            ax.set_xlim([-2.05,2.05])
                            ax.set_ylim([-2.05,2.05])            
                elif row != rows - 1:
                    ax = f.add_subplot(spec[row, col])
                    cl = col - 1
                    print(row, cl)                    
                    u, s, v = self.calc_point_SVD(marks[cl, 0], marks[cl, 1], marks[cl, 2])
                    if np.max(u[:,row]) > 0:
                        sh = shcoeffs.SHCoeffs(u[:,row])
                        shfilename = folder+'/'+str(row)+str(cl)+'.png'
                        sh.plot_dist(filename=shfilename, r=1.1)
                        from PIL import Image
                        im1 = np.asarray(Image.open(shfilename))
                        ax.imshow(im1, interpolation='none')
                        ax.annotate('$\sigma='+'{:.2f}'.format(s[row]/self.sigma_max)+'$', xy=(1, 1), xytext=(0.5, 0),
                                    textcoords='axes fraction', ha='center', va='center')

                    ax.axis('off')
                elif row == rows - 1 and col == 0:
                    ax = f.add_subplot(spec[row, col])
                    # Colorbars
                    X, Y = np.meshgrid(np.linspace(0, 1, 100),
                                       np.linspace(0, 1, 100))
                    ax.imshow(X, cmap="bwr", vmin=-1, vmax=1, interpolation='none', extent=[0,1,0,1], origin='lower', aspect='auto')
                    # ax.contour(X, levels, colors='k',linewidths=0.5, extent=[0,1,0,1], origin='lower',)
                    ax.set_xlim([0,1])
                    ax.set_ylim([0,1])
                    ax.tick_params(direction='out', bottom=True, top=False)
                    ax.xaxis.set_ticks([0, 0.5, 1.0])
                    ax.yaxis.set_ticks([])
                    
        f.savefig(filename, bbox_inches='tight')
        
    def plot_scene(self, filename):
        print('Plotting: ' + filename)
        scene_string = ''
        for m in self.micros:
            scene_string += m.scene_string()
        util.draw_scene(scene_string, filename=filename, save_file=True)
        
    def plot_frames(self, folder='out'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, m in enumerate(self.micros):
            m.plot_scene(filename=folder+'/scene'+str(i)+'.pdf')
            m.plot(m.H, filename=folder+'/otf'+str(i)+'.pdf')
