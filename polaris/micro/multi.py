from polaris import util, viz, data, spang
from polaris.micro import ill, det, micro
from polaris.harmonics import shcoeffs
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import logging
log = logging.getLogger('log')

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
    def __init__(self, spang, data, sigma_ax=0.25, n_samp=1.33, lamb=525,
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
        out = np.outer(tf[0,:], np.ones(pols.shape[1])) + np.outer(tf[1,:], -2*x*y) + np.outer(tf[2,:], x**2 - y**2)

        return out # return s x p

    def calc_H(self):
        # Transverse transfer function
        log.info('Computing H for view 0')
        dx = np.fft.rfftfreq(self.X, d=self.data.vox_dim[0])*self.lamb/self.micros[0].det.na
        dy = np.fft.rfftfreq(self.Y, d=self.data.vox_dim[1])*self.lamb/self.micros[0].det.na
        dz = np.fft.rfftfreq(self.Z, d=self.data.vox_dim[2])*self.lamb/self.micros[0].det.na
        self.Hxy = np.zeros((dx.shape[0], dy.shape[0], self.J, self.P), dtype=np.float32)
        for x, nux in enumerate(tqdm(dx)):
            for y, nuy in enumerate(dy):
                self.Hxy[x,y,:,:] = self.calc_point_H(nux, nuy, 0, 0, self.data.pols_norm)
        self.Hxy = self.Hxy/np.max(np.abs(self.Hxy))
        if self.micros[0].spang_coupling:
            self.Hz = np.exp(-(dz**2)/(2*(self.sigma_ax**2)), dtype=np.float32)
        else:
            self.Hz = np.ones(dz.shape, dtype=np.float32)

        log.info('Computing H for view 1')
        dx = np.fft.rfftfreq(self.X, d=self.data.vox_dim[0])*self.lamb/self.micros[1].det.na
        dy = np.fft.rfftfreq(self.Y, d=self.data.vox_dim[1])*self.lamb/self.micros[1].det.na
        dz = np.fft.rfftfreq(self.Z, d=self.data.vox_dim[2])*self.lamb/self.micros[1].det.na
        self.Hyz = np.zeros((dy.shape[0], dz.shape[0], self.J, self.P), dtype=np.float32)
        for y, nuy in enumerate(tqdm(dy)):
            for z, nuz in enumerate(dz):
                self.Hyz[y,z,:,:] = self.calc_point_H(0, nuy, nuz, 1, self.data.pols_norm)
        self.Hyz = self.Hyz/np.max(np.abs(self.Hyz))
        if self.micros[0].spang_coupling:
            self.Hx = np.exp(-(dx**2)/(2*(self.sigma_ax**2)))
        else:
            self.Hx = np.ones(dx.shape)

    def lake_response(self, data):
        e0 = self.calc_point_H(0, 0, 0, 0, data.pols_norm)[0,:]
        e1 = self.calc_point_H(0, 0, 0, 1, data.pols_norm)[0,:]
        return np.vstack([e0, e1])
            
    def save_H(self, filename):
        np.savez(filename, Hxy=self.Hxy, Hyz=self.Hyz, Hx=self.Hx, Hz=self.Hz)
        
    def load_H(self, filename):
        files = np.load(filename)
        self.Hxy = files['Hxy']
        self.Hyz = files['Hyz']
        self.Hx = files['Hx']        
        self.Hz = files['Hz']
        
    def fwd(self, f, snr=None):
        log.info('Applying forward operator')

        # Truncate spang for angular bandlimit
        f = f[:,:,:,:self.jmax]

        # 3D FT
        F = np.fft.rfftn(f, axes=(0,1,2))
        
        # Tensor multiplication
        G = np.zeros(F.shape[0:3] + (self.P,) + (self.V,), dtype=np.complex64)
        for x in tqdm(range(self.Hxy.shape[0])):
            for y in range(self.Hxy.shape[1]):
                Hzsp = np.einsum('z,sp->zsp', self.Hz, self.Hxy[x,y,:,:])
                G[x,y,:,:,0] = np.einsum('zsp,zs->zp', Hzsp, F[x,y,:,:])
                G[-x,y,:,:,0] = np.einsum('zsp,zs->zp', Hzsp, F[-x,y,:,:])
                G[x,-y,:,:,0] = np.einsum('zsp,zs->zp', Hzsp, F[x,-y,:,:])
                G[-x,-y,:,:,0] = np.einsum('zsp,zs->zp', Hzsp, F[-x,-y,:,:])

        start = slice(0, (self.X//2)+1)
        end = slice(None, -(self.X//2), -1)
        for y in tqdm(range(self.Hyz.shape[0])):
            for z in range(self.Hyz.shape[1]):
                Hxsp = np.einsum('x,sp->xsp', self.Hx, self.Hyz[y,z,:,:])
                G[start,y,z,:,1] = np.einsum('xsp,xs->xp', Hxsp, F[start,y,z,:])
                G[end,y,z,:,1] = np.einsum('xsp,xs->xp', Hxsp[1:-1], F[end,y,z,:])
                G[start,-y,z,:,1] = np.einsum('xsp,xs->xp', Hxsp, F[start,-y,z,:])
                G[end,-y,z,:,1] = np.einsum('xsp,xs->xp', Hxsp[1:-1], F[end,-y,z,:])

        # 3D IFT
        g = np.fft.irfftn(G, s=f.shape[0:3], axes=(0,1,2))
        
        # Apply Poisson noise
        if snr is not None:
            norm = snr**2/np.max(g)
            arr_poisson = np.vectorize(np.random.poisson)
            g = arr_poisson(g*norm)/norm

        # return g
        return g/np.max(g)
    
    def fwd_angular(self, f, snr=None, mask=None):
        log.info('Applying angular forward operator')
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

    def pinv(self, g, eta=0, padding=True):
        # 3D FT
        log.info('Taking 3D Fourier transform')        
        G = np.fft.rfftn(g, axes=(0,1,2))
        G2 = np.reshape(G, G.shape[0:3] + (self.P*self.V,))
        
        xstart = slice(0, (self.X//2)+1)
        xend = slice(None, -(self.X//2), -1)
        ystart = slice(0, (self.Y//2)+1)
        yend = slice(None, -(self.Y//2), -1)

        log.info('Applying pseudoinverse operator')
        import multiprocessing as mp
        pool = mp.Pool(processes=mp.cpu_count())
        args = []
        #import pdb; pdb.set_trace()
        for z in range(self.Hz.shape[0]):
            args.append((z, G2[:,:,z,:], self.Hxy, self.Hyz[:,z,:,:], self.Hx, self.Hz, self.X, self.Y, self.J, self.P, self.V, eta, xstart, xend, ystart, yend))
        result = list(tqdm(pool.imap(compute_pinv, args), total=len(args)))
        F = np.array(result)
        del result
        
        # 3D IFT
        log.info('Taking inverse 3D Fourier transform')        
        f = np.fft.irfftn(np.moveaxis(F, 0, 2), s=g.shape[0:3], axes=(0,1,2))

        return f
    
    def pinv_angular(self, g, eta=0, mask=None):
        log.info('Applying pseudoinverse operator')
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

def compute_pinv(args):
    z, G2, Hxy, Hyz, Hx, Hz, X, Y, J, P, V, eta, xstart, xend, ystart, yend = args
    H0 = Hz[z]*Hxy[:,:,:,:]
    H1 = np.einsum('x,ysp->xysp', Hx, Hyz[:,:,:])
    HH = np.reshape(np.stack((H0, H1), axis=-1), H0.shape[0:2] + (J, P*V))
    u, s, vh = np.linalg.svd(HH, full_matrices=False) # Find SVD
    sreg = np.where(s > 1e-7, s/(s**2 + eta), 0) # Regularize
    Pinv = np.einsum('xysv,xyv,xyvd->xysd', u, sreg, vh, optimize=True)
    F = np.zeros((X, Y, J), dtype=np.complex64)
    F[xstart,ystart,:] = np.einsum('xysd,xyd->xys', Pinv[:,:,:,:], G2[xstart,ystart,:])
    F[xend,ystart,:] = np.einsum('xysd,xyd->xys', Pinv[1:-1,:,:,:], G2[xend,ystart,:])
    F[xstart,yend,:] = np.einsum('xysd,xyd->xys', Pinv[:,1:-1,:,:], G2[xstart,yend,:])
    F[xend,yend,:] = np.einsum('xysd,xyd->xys', Pinv[1:-1,1:-1,:,:], G2[xend,yend,:])
    return F
