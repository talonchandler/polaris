from polaris import spang, util
import numpy as np
from scipy.special import hyp1f1
from dipy.data import get_sphere
import logging
log = logging.getLogger('log')

def three_helix(vox_dim=(130,130,130), px=(64,64,64), radius=600):
    phant = spang.Spang(np.zeros(px + (15,), dtype=np.float32), vox_dim=vox_dim)
    s = 20
    phant1 = helix_phantom(px=(s,s,s), radius=radius, pitch=1000,
                          vox_dim=vox_dim, max_l=4, center=(0,0,0),
                          normal=0, trange=(-2*np.pi, 2*np.pi))
    phant2 = helix_phantom(px=(s,s,s), radius=radius, pitch=1000,
                          vox_dim=vox_dim, max_l=4, center=(0,0,0),
                          normal=1, trange=(-2*np.pi, 2*np.pi))
    phant3 = helix_phantom(px=(s,s,s), radius=radius, pitch=1000,
                          vox_dim=vox_dim, max_l=4, center=(0,0,0),
                          normal=2, trange=(-2*np.pi, 2*np.pi))
    phant.f[0:s,0:s,0:s] = phant1.f
    phant.f[s:2*s,s:2*s,s:2*s] = phant2.f
    phant.f[2*s:3*s,2*s:3*s,2*s:3*s] = phant3.f
    return phant

def helix_phantom(px=(20,20,20), vox_dim=(100,100,100), max_l=6,
                  trange=(-4*np.pi, 4*np.pi), nt=100, radius=700, pitch=1000,
                  cyl_rad=250, center=(0,0,0), normal=0, krange=(0,5),
                  dtype=np.float32):
    log.info('Generating helix')
    t = np.linspace(trange[0], trange[1], nt)
    c = np.array([radius*np.cos(t), radius*np.sin(t), pitch*t/(2*np.pi)]).T
    d = np.array([-radius*np.sin(t), radius*np.cos(t), pitch/(2*np.pi) + 0*t]).T
    d = d/np.linalg.norm(d, axis=-1)[...,None] # normalize
    c = np.roll(c, normal, axis=-1) + center # orient and recenter
    d = np.roll(d, normal, axis=-1) # orient
    k = np.linspace(krange[0], krange[1], nt) # watson kappa parameter
    return curve_phantom(c, d, k, cyl_rad=cyl_rad, vox_dim=vox_dim, px=px, max_l=max_l, dtype=dtype)

def curve_phantom(curve, direction, kappa,
                  px=(20,20,20), vox_dim=(100,100,100), cyl_rad=0.2, max_l=6,
                  dtype=np.float32):

    # Setup grid
    xyz = np.array(np.meshgrid(
        np.linspace(-(px[0]/2)*vox_dim[0], (px[0]/2)*vox_dim[0], px[0]),
        np.linspace(-(px[1]/2)*vox_dim[1], (px[1]/2)*vox_dim[1], px[1]),
        np.linspace(-(px[2]/2)*vox_dim[2], (px[2]/2)*vox_dim[2], px[2])))
    xyz = np.moveaxis(xyz, [0, 1], [-1, 1])

    # Calculate directions and kappas
    diff = xyz[:,:,:,None,:] - curve
    dist = np.linalg.norm(diff, axis=-1)
    min_dist = np.min(dist, axis=-1) # min dist between grid points and curve
    t_index = np.argmin(dist, axis=-1)
    min_dir = direction[t_index] # directions for closest point on curve
    min_k = kappa[t_index] # kappas
    
    # Calculate watson 
    spang_shape = xyz.shape[0:-1] + (util.maxl2maxj(max_l),)
    spang1 = spang.Spang(np.zeros(spang_shape), vox_dim=vox_dim)
    dot = np.einsum('ijkl,ml->ijkm', min_dir, spang1.sphere.vertices)
    k = min_k[...,None]
    watson = np.exp(k*dot**2)/(4*np.pi*hyp1f1(0.5, 1.5, k))
    watson_sh = np.einsum('ijkl,lm', watson, spang1.B)
    watson_sh = watson_sh/watson_sh[...,None,0] # Normalize

    # Cylinder mask
    mask = min_dist < cyl_rad
    spang1.f = np.einsum('ijkl,ijk->ijkl', watson_sh, mask).astype(dtype)
    
    return spang1

def bead(orientation=[1,0,0], kappa=None, px=(32,32,32), vox_dim=(100,100,100)):
    # orientation sets the axis of rotational symmetry
    # kappa = None: all along single axis
    # kappa = 0: angularly uniform
    # kappa > 0: dumbbell
    # kappa < 0: pancake
    dims = px + (15,)
    f = np.zeros(dims, dtype=np.float32)
    spang1 = spang.Spang(f, vox_dim=vox_dim)
    dot = np.dot(orientation, spang1.sphere.vertices.T)
    if kappa is None:
        spang1.f[px[0]//2, px[1]//2, px[2]//2, :] = util.xyz_sft(orientation, max_l=4)
    else:
        watson = np.exp(kappa*dot**2)/(4*np.pi*hyp1f1(0.5, 1.5, kappa))
        watson_sh = np.matmul(spang1.B.T, watson)
        spang1.f[px[0]//2, px[1]//2, px[2]//2, :] = watson_sh
    
    return spang1

def all_directions(px=(15,15,1), vox_dim=(100,100,100)):
    dims = px + (3,)
    # f = np.zeros(dims)
    # tp = np.array(np.meshgrid(np.linspace(0, np.pi/2, px[0]),
    #                           np.linspace(0, np.pi, px[1]),
    #                           np.linspace(0, 0, px[2])))
    # tp = np.moveaxis(tp, [0, 1], [-1, 1])

    # xyz = np.apply_along_axis(util.tp2xyz, -1, tp)
    xyz = util.fibonacci_sphere(np.prod(np.array(px)), xyz=True).reshape(dims)
    f = np.apply_along_axis(util.xyz_sft, -1, xyz, max_l=4)

    return spang.Spang(f, vox_dim=vox_dim)
