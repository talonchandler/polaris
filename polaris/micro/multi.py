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
    def __init__(self, spang, data, sigma_ax=0.33, n_samp=1.33, lamb=525,
                 spang_coupling=True):

        self.spang = spang
        self.data = data

        self.X = spang.X
        self.Y = spang.Y
        self.Z = spang.Z
        self.J = spang.J
        self.P = data.P                
        self.V = data.V
        
        m = [] # List of microscopes

        # Cycle through paths
        for i, det_optical_axis in enumerate(data.det_optical_axes):
            ill_ = ill.Illuminator(optical_axis=data.ill_optical_axes[i],
                                   na=data.ill_nas[i], n=n_samp)
            det_ = det.Detector(optical_axis=data.det_optical_axes[i],
                                na=data.det_nas[i], n=n_samp, sigma_ax=sigma_ax)
            m.append(micro.Microscope(ill=ill_, det=det_, spang_coupling=spang_coupling)) # Add microscope

        self.micros = m
        self.lamb = lamb
        self.sigma_ax = sigma_ax
        self.jmax = m[0].h(0, 0, 0).jmax

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

        return out # return s x p

    def calc_H(self):
        Hxy = np.zeros((self.X, self.Y, self.J, self.P, self.V), dtype=np.float32)

        print('Computing H')
        for v in range(self.V):
            dx = np.fft.fftshift(np.fft.fftfreq(self.X, d=self.data.vox_dim[0]))*self.lamb/self.micros[v].det.na
            dy = np.fft.fftshift(np.fft.fftfreq(self.Y, d=self.data.vox_dim[1]))*self.lamb/self.micros[v].det.na
            for x, nux in enumerate(tqdm(dx)):
                for y, nuy in enumerate(dy):
                    if v == 0:
                        Hxy[x,y,:,:,v] = self.calc_point_H(nux, nuy, 0, v, self.data.pols_norm)
                    if v == 1: 
                        Hxy[x,y,:,:,v] = self.calc_point_H(0, nuy, nux, v, self.data.pols_norm)
        self.Hxy = Hxy/np.max(np.abs(Hxy))

        dz = np.fft.fftshift(np.fft.fftfreq(self.Z, d=self.data.vox_dim[2]))*self.lamb/self.micros[v].det.na
        self.Hz = np.exp(-(dz**2)/(2*(self.sigma_ax**2)))

        # Shift H instead of shifting F and G  
        self.Hxy = np.fft.fftshift(self.Hxy, axes=(0,1))
        self.Hz = np.fft.fftshift(self.Hz)

    def save_H(self, filename):
        np.savez(filename, Hxy=self.Hxy, Hz=self.Hz)
        
    def load_H(self, filename):
        files = np.load(filename)
        self.Hxy = files['Hxy']
        self.Hz = files['Hz']
        
    def fwd(self, f, snr=None):
        print('Applying forward operator')

        # Truncate spang for angular bandlimit
        f = f[:,:,:,:self.jmax]
        
        # 3D FT
        F = np.fft.fftn(f, axes=(0,1,2))
        
        # Tensor multiplication
        G = np.zeros(self.data.g.shape, dtype=np.complex64)
        for x in tqdm(range(self.X)):
            for y in range(self.Y):
                for z in range(self.Z):
                    G[x,y,z,:,0] = self.Hz[z]*np.einsum('sp,s->p', self.Hxy[x,y,:,:,0], F[x,y,z])
                    G[x,y,z,:,1] = self.Hz[x]*np.einsum('sp,s->p', self.Hxy[y,z,:,:,1], F[x,y,z])
            
        # 3D IFT
        g = np.real(np.fft.ifftn(G, axes=(0,1,2)))

        # Apply Poisson noise
        if snr is not None:
            norm = snr**2/np.max(g)
            arr_poisson = np.vectorize(np.random.poisson)
            g = arr_poisson(g*norm)/norm

        # return g
        return g/np.max(g)

    def fwd_angular(self, f, snr=None, mask=None):
        print('Applying angular forward operator')
        g = np.zeros(self.data.g.shape)
        H = np.zeros((self.J, self.P, self.V))
        H[:,:,0] = self.calc_point_H(0, 0, 0, 0, self.data.pols_norm)
        H[:,:,1] = self.calc_point_H(0, 0, 0, 1, self.data.pols_norm)

        # Setup mask
        if mask is None:
            mask = np.ones((self.X, self.Y, self.Z))

        # Main calculation
        for x in tqdm(range(self.X)):
            for y in range(self.Y):
                for z in range(self.Z):
                    if mask[x,y,z]:
                        g[x,y,z,:,:] = np.einsum('spv,s->pv', H, f[x,y,z,:])
        return g/np.max(g)
    
    def pinv(self, g, eta=0):
        print('Applying pseudoinverse operator')
        
        # 3D FT
        G = np.fft.fftn(g, axes=(0,1,2))
        G2 = np.reshape(G, (self.X, self.Y, self.Z, self.P*self.V))
        
        # Tensor multiplication
        F = np.zeros((self.X, self.Y, self.Z, self.J), dtype=np.complex64)
        for x in tqdm(range(self.X)):
            for y in range(self.Y):
                for z in range(self.Z):
                    # Generate H and order it properly
                    H0 = self.Hz[z]*self.Hxy[x,y,:,:,0] 
                    H1 = self.Hz[x]*self.Hxy[y,z,:,:,1]
                    HH = np.reshape(np.stack((H0, H1), axis=-1), (self.J, self.P*self.V)) 
                    u, s, vh = np.linalg.svd(HH, full_matrices=False) # Find SVD
                    sreg = np.where(s > 1e-7, s/(s**2 + eta), 0) # Regularize
                    F[x,y,z,:] = np.einsum('lm,m,mn,n->l', u, sreg, vh, G2[x,y,z,:]) # Apply Pinv
        
        # 3D IFT        
        f = np.real(np.fft.ifftn(F, axes=(0,1,2)))

        return f

    def pinv_angular(self, g, eta=0, mask=None):
        print('Applying pseudoinverse operator')
        f = np.zeros(self.spang.f.shape)
        H = np.zeros((self.J, self.P, self.V))
        H[:,:,0] = self.calc_point_H(0, 0, 0, 0, self.data.pols_norm)
        H[:,:,1] = self.calc_point_H(0, 0, 0, 1, self.data.pols_norm)
        HH = np.reshape(H, (self.J, self.P*self.V))
        u, s, vh = np.linalg.svd(HH, full_matrices=False) # Find SVD
        sreg = np.where(s > 1e-7, s/(s**2 + eta), 0) # Regularize
        M = np.einsum('lm,m,mn->ln', u, sreg, vh)

        # Setup mask
        if mask is None:
            mask = np.ones((self.X, self.Y, self.Z))

        gg = np.reshape(g, (self.X, self.Y, self.Z, self.P*self.V))

        # Main calculation
        for x in tqdm(range(self.X)):
            for y in range(self.Y):
                for z in range(self.Z):
                    if mask[x,y,z]:
                        f[x,y,z,:] = np.einsum('ln,n->l', M, gg[x,y,z,:]) # Apply Pinv
        return f
        
    def pmeas(self, f, eta=0):
        return self.pinv(self.fwd(f), eta=eta)

    def pnull(self, f):
        return f - self.pmeas(f) # could be made more efficient
