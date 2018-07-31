from polaris import spang, util
import numpy as np

def curve_phantom(curve, direction, px=(20,20,20), vox_dim=(100,100,100),
                  cyl_rad=0.2, max_l=6):
    xyz = np.array(np.meshgrid(
        np.linspace(-(px[0]/2)*vox_dim[0], (px[0]/2)*vox_dim[0], px[0]),
        np.linspace(-(px[1]/2)*vox_dim[1], (px[1]/2)*vox_dim[1], px[1]),
        np.linspace(-(px[2]/2)*vox_dim[2], (px[2]/2)*vox_dim[2], px[2])))
    xyz = np.moveaxis(xyz, [0, 1], [-1, 1])

    diff = xyz[:,:,:,None,:] - curve
    dist = np.linalg.norm(diff, axis=-1)
    min_dist = np.min(dist, axis=-1) # min dist between grid points and curve
    t_index = np.argmin(dist, axis=-1)
    min_d = direction[t_index] # directions for closest point on curve

    # Apply mask
    mask = min_dist < cyl_rad
    odf = np.einsum('ijkl,ijk->ijkl', min_d, mask)
    
    f = np.apply_along_axis(util.xyz_sft, 3, odf, max_l=max_l)
    return spang.Spang(f, vox_dim=vox_dim)

def helix_phantom(nt=100, radius=700, pitch=1000, cyl_rad=250,
                  px=(20,20,20), vox_dim=(100,100,100), max_l=6):
    t = np.linspace(-2*np.pi, 2*np.pi, nt)
    c = np.array([radius*np.cos(t), radius*np.sin(t), pitch*t/(2*np.pi)]).T
    d = np.array([-radius*np.sin(t), radius*np.cos(t), pitch/(2*np.pi) + 0*t]).T
    return curve_phantom(c, d, cyl_rad=cyl_rad, vox_dim=vox_dim, px=px, max_l=max_l)

def bead(orientation=[1,0,0], uniform=False, px=(21,21,21),
                 vox_dim=(100,100,100)):
    dims = px + (15,)
    f = np.zeros(dims)
    if uniform:
        f[int((px[0]-1)/2), int((px[1]-1)/2), int((px[2]-1)/2), 0] = 1
    else:
        f[int((px[0]-1)/2), int((px[1]-1)/2), int((px[2]-1)/2), :] = util.xyz_sft(orientation, max_l=4)
    
    return spang.Spang(f, vox_dim=vox_dim)

def all_directions(px=(15,15,1), vox_dim=(100,100,100)):
    dims = px + (3,)
    f = np.zeros(dims)
    tp = np.array(np.meshgrid(np.linspace(0, np.pi/2, px[0]),
                              np.linspace(0, np.pi, px[1]),
                              np.linspace(0, 0, px[2])))
    tp = np.moveaxis(tp, [0, 1], [-1, 1])

    xyz = np.apply_along_axis(util.tp2xyz, -1, tp)
    f = np.apply_along_axis(util.xyz_sft, -1, xyz, max_l=4)
    return spang.Spang(f, vox_dim=vox_dim)
