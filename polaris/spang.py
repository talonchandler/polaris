import subprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('text', usetex=True)
from polaris import viz, util
import numpy as np
from dipy.viz import window, actor
from dipy.data import get_sphere
import vtk
from tqdm import tqdm
import tifffile
import os

class Spang:
    """
    A Spang (short for spatio-angular density) is a representation of a 
    spatio-angular density f(r, s) stored as a 4D array of voxel values 
    and spherical harmonic coefficients [x, y, z, j]. A Spang object is 
    a discretized member of object space U. 
    """
    def __init__(self, f=np.zeros((3,3,3,1)), vox_dim=(1,1,1),
                 sphere=get_sphere('symmetric724')):
        self.X = f.shape[0]
        self.Y = f.shape[1]
        self.Z = f.shape[2]
        
        # Calculate band dimensions
        self.lmax, mm = util.j2lm(f.shape[-1] - 1)
        self.J = util.maxl2maxj(self.lmax)

        # Fill the rest of the last l band with zeros
        if f.shape[-1] != self.J:
            temp = np.zeros((self.X, self.Y, self.Z, self.J))
            temp[...,:f.shape[-1]] = f
            self.f = temp
        else:
            self.f = f

        self.vox_dim = vox_dim
        self.sphere = sphere
        self.N = len(self.sphere.theta)
        self.calc_B()
        
    def calc_B(self):
        # Calculate odf to sh matrix
        B = np.zeros((self.N, self.J))
        for (n, j), x in np.ndenumerate(B):
            l, m = util.j2lm(j)
            B[n, j] = util.spZnm(l, m, self.sphere.theta[n], self.sphere.phi[n])
        self.B = B
        self.Binv = np.linalg.pinv(self.B, rcond=1e-15)

    def density(self):
        return self.f[...,0]

    def gfa(self):
        return np.nan_to_num(np.sqrt(1 - (self.f[...,0]**2)/np.sum(self.f**2, axis=-1)))

    def tensor(self):
        print("Calculating tensor fits")
        M = np.load(os.path.join(os.path.dirname(__file__), 'harmonics/sh2tensor.npy'))
        Di = np.einsum('ijkl,lm->ijkm', self.f[...,0:6], M)
        D = np.zeros(self.f.shape[0:3]+(3,3), dtype=np.float32)
        D[...,0,0] = Di[...,0]; D[...,0,1] = Di[...,3]; D[...,0,2] = Di[...,5];
        D[...,1,0] = Di[...,3]; D[...,1,1] = Di[...,1]; D[...,1,2] = Di[...,4];
        D[...,2,0] = Di[...,5]; D[...,2,1] = Di[...,4]; D[...,2,2] = Di[...,2];
        eigs = np.linalg.eigh(D)
        principal = eigs[1][...,-1]*eigs[1][...,-1]
        return Di.astype(np.float32), principal.astype(np.float32)
        
    def save_summary(self, filename='out.pdf', gfa_filter=0.1, mag=4,
                     mask=None, scale=1, keep_parallels=False, skip_n=1):
        print('Generating ' + filename)
        pos = (-0.05, 1.05, 0.5, 0.55) # Arrow and label positions
        vmin = 0
        vmax = 1
        inches = 2
        rows = 2
        cols = 3
        colormap = 'Reds'
        widths = [1]*cols
        heights = [1]*rows
        M = np.max(self.f.shape)
        x_frac = self.f.shape[0]/M
        if gfa_filter is None:
            gfa_label = 'GFA'
        else:
            gfa_label = 'GFA where density $>$ ' + str(gfa_filter)
        if skip_n == 1:
            ds_string = ''
        else:
            ds_string = 'downsampled ' + str(skip_n) + '$\\times$'
        col_labels = np.array([['Peak' + ds_string, 'Ellipsoid' + ds_string, 'Principal' + ds_string], ['ODF' + ds_string, 'Normalized density' + ds_string, gfa_label]])
        f = plt.figure(figsize=(inches*np.sum(widths), inches*np.sum(heights)))
        spec = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths,
                                 height_ratios=heights, hspace=0.075, wspace=0.075)
        for row in range(rows):
            for col in range(cols):
                if col < 3:
                    yscale_label = None
                    if row == 1 and col == 0:
                        bar = True
                        bar_label = 'ODF radius'
                        colormap = 'Reds'
                        self.visualize(out_path='parallels/', zoom=1.7,
                                       outer_box=False, axes=False,
                                       clip_neg=False, azimuth=0, elevation=0,
                                       n_frames=1, mag=mag, video=False, scale=scale,
                                       interact=False, viz_type='ODF',
                                       save_parallels=True, mask=mask, skip_n=skip_n)
                    if row == 0 and col == 1:
                        bar = False
                        bar_label = 'Principal'
                        self.visualize(out_path='parallels/', zoom=1.7,
                                       outer_box=False, axes=False,
                                       clip_neg=False, azimuth=0, elevation=0,
                                       n_frames=1, mag=mag, video=False, scale=scale,
                                       interact=False, viz_type='ELLIPSOID',
                                       save_parallels=True, mask=mask, skip_n=skip_n)
                    if row == 0 and col == 2:
                        bar = False
                        bar_label = 'Principal'
                        self.visualize(out_path='parallels/', zoom=1.7,
                                       outer_box=False, axes=False,
                                       clip_neg=False, azimuth=0, elevation=0,
                                       n_frames=1, mag=mag, video=False, scale=scale,
                                       interact=False, viz_type='PRINCIPAL',
                                       save_parallels=True, mask=mask, skip_n=skip_n)
                    if row == 0 and col == 0:
                        bar = False
                        bar_label = 'Peak'
                        self.visualize(out_path='parallels/', zoom=1.7,
                                       outer_box=False, axes=False,
                                       clip_neg=False, azimuth=0, elevation=0,
                                       n_frames=1, mag=mag, video=False, scale=scale,
                                       interact=False, viz_type='PEAK',
                                       save_parallels=True, mask=mask, skip_n=skip_n)
                    if row == 1 and col == 1:
                        colormap = 'gray'
                        bar = True
                        bar_label = 'Normalized\n density'
                        viz.plot_parallels(self.density(), out_path='parallels/', outer_box=False,
                                           axes=False, clip_neg=False, azimuth=0,
                                           elevation=0, scale=scale)
                    if row == 1 and col == 2:
                        self.yscale = 1e-3*self.vox_dim[1]*self.f.shape[0]
                        colormap = 'gray'
                        yscale_label = '{:.2f}'.format(self.yscale) + ' $\mu$m'
                        bar = True
                        bar_label = 'GFA'
                        viz.plot_parallels(self.gfa(), out_path='parallels/', outer_box=False,
                                           axes=False, clip_neg=False, azimuth=0,
                                           elevation=0, scale=scale, mask=self.density() > gfa_filter)

                    viz.plot_images(['parallels/yz.png', 'parallels/xy.png', 'parallels/xz.png'],
                                    f, spec, row, col,
                                    col_labels=col_labels, row_labels=None,
                                    vmin=vmin, vmax=vmax, colormap=colormap,
                                    rows=rows, cols=cols, x_frac=x_frac,
                                    yscale_label=yscale_label, pos=pos, bar=bar, bar_label=bar_label)
                    if not keep_parallels:
                        subprocess.call(['rm', '-r', 'parallels'])
                    
                elif col == 3:
                    viz.plot_colorbar(f, spec, row, col, vmin, vmax, colormap)

        print('Saving ' + filename)
        # plt.subplots_adjust(bottom=0.3)
        f.savefig(filename, bbox_inches='tight')
        
    def visualize(self, out_path='out/', outer_box=True, axes=True,
                  clip_neg=False, azimuth=0, elevation=0, n_frames=1,
                  size=(600,600), mag=4, video=False, viz_type='ODF', mask=None,
                  skip_n=1, scale=1, zoom=1.0, zoom_in=1.0, interact=False,
                  save_parallels=False, gfa_filter=0):
        print('Preparing to render ' + out_path)
        
        # Prepare output
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        # Mask
        if mask is None:
            mask = np.ones((self.X, self.Y, self.Z), dtype=np.bool)
        for x in [-1,0]:
            for y in [-1,0]:
                for z in [-1,0]:
                    mask[x,y,z] = True
        skip_mask = np.zeros(mask.shape, dtype=np.bool)
        skip_mask[::skip_n,::skip_n,::skip_n] = 1
        mask = np.logical_and(mask, skip_mask)

        # Render
        ren = window.Renderer()
        ren.background([1,1,1])

        # Add visuals to renderer
        if viz_type == "ODF":
            print('Rendering '+str(np.sum(mask) - 8)+' ODFs')
            fodf_spheres = viz.odf_sparse(self.f, self.Binv, sphere=self.sphere,
                                          scale=skip_n*scale*0.5, norm=False,
                                          colormap='bwr', mask=mask,
                                          global_cm=True)
            ren.add(fodf_spheres)
        elif viz_type == "ELLIPSOID":
            print('Rendering '+str(np.sum(mask) - 8)+' ellipsoids')
            fodf_peaks = viz.tensor_slicer_sparse(self.f,
                                                  sphere=self.sphere,
                                                  scale=skip_n*scale*0.5,
                                                  mask=mask)
            ren.add(fodf_peaks)
        elif viz_type == "PEAK":
            print('Rendering '+str(np.sum(mask) - 8)+' peaks')
            fodf_peaks = viz.peak_slicer_sparse(self.f, self.Binv, self.sphere.vertices, 
                                                scale=skip_n*scale*0.5,
                                                mask=mask)
            ren.add(fodf_peaks)
        elif viz_type == "PRINCIPAL":
            print('Rendering '+str(np.sum(mask) - 8)+' principals')
            fodf_peaks = viz.principal_slicer_sparse(self.f, self.Binv, self.sphere.vertices, 
                                                     scale=skip_n*scale*0.5,
                                                     mask=mask)
            ren.add(fodf_peaks)

        X = self.X - 1
        Y = self.Y - 1
        Z = self.Z - 1

        # Add invisible actors to set FOV
        NN = np.max([X, Y, Z])
        ren.add(actor.line([np.array([[0,0,0],[NN,0,0]])], colors=np.array([1,1,1]), linewidth=1))
        ren.add(actor.line([np.array([[0,0,0],[0,NN,0]])], colors=np.array([1,1,1]), linewidth=1))
        ren.add(actor.line([np.array([[0,0,0],[0,0,NN]])], colors=np.array([1,1,1]), linewidth=1))
        if outer_box:
            c = np.array([0,0,0])
            ren.add(actor.line([np.array([[0,0,0],[X,0,0],[X,Y,0],[0,Y,0],
                                          [0,0,0],[0,Y,0],[0,Y,Z],[0,0,Z],
                                          [0,0,0],[X,0,0],[X,0,Z],[0,0,Z]])], colors=c))
            ren.add(actor.line([np.array([[X,0,Z],[X,Y,Z],[X,Y,0],[X,Y,Z],
                                          [0,Y,Z]])], colors=c))
        # Add colored axes
        if axes:
            ren.add(actor.line([np.array([[0,0,0],[NN/10,0,0]])], colors=np.array([1,0,0]), linewidth=4))
            ren.add(actor.line([np.array([[0,0,0],[0,NN/10,0]])], colors=np.array([0,1,0]), linewidth=4))
            ren.add(actor.line([np.array([[0,0,0],[0,0,NN/10]])], colors=np.array([0,0,1]), linewidth=4))

        # Setup vtk renderers
        renWin = vtk.vtkRenderWindow()
        if not interact:
            renWin.SetOffScreenRendering(1)
        renWin.AddRenderer(ren)
        renWin.SetSize(size[0], size[1])
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        ren.ResetCamera()
        ren.azimuth(azimuth)
        ren.elevation(elevation)
        
        writer = vtk.vtkPNGWriter()
        az = 0
        naz = np.ceil(360/n_frames)
        
        print('Rendering ' + out_path)
        if save_parallels:
            filenames = ['yz', 'xy', 'xz']
            zooms = [zoom, 1.0, 1.0]
            azs = [90, -90, 0]
            els = [0, 0, 90]
            for i in tqdm(range(3)):
                ren.projection(proj_type='parallel')
                ren.zoom(zooms[i])
                ren.azimuth(azs[i])
                ren.elevation(els[i])
                ren.reset_clipping_range()
                renderLarge = vtk.vtkRenderLargeImage()
                renderLarge.SetMagnification(mag)
                renderLarge.SetInput(ren)
                renderLarge.Update()
                writer.SetInputConnection(renderLarge.GetOutputPort())
                writer.SetFileName(out_path + filenames[i] + '.png')
                writer.Write()
        else:
            ren.zoom(zoom)
            for i in tqdm(range(n_frames)):
                ren.zoom(1 + ((zoom_in - zoom)/n_frames))
                ren.azimuth(az)
                ren.reset_clipping_range()
                renderLarge = vtk.vtkRenderLargeImage()
                renderLarge.SetMagnification(mag)
                renderLarge.SetInput(ren)
                renderLarge.Update()
                writer.SetInputConnection(renderLarge.GetOutputPort())
                writer.SetFileName(out_path + str(i).zfill(6) + '.png')
                writer.Write()
                az = naz

        # Generate video (requires ffmpeg)
        if video:
            print('Generating video from frames')
            fps = np.ceil(n_frames/12)
            subprocess.call(['ffmpeg', '-nostdin', '-y', '-framerate', str(fps),
                             '-loglevel', 'panic',
                             '-i', out_path+'%06d.png', out_path[:-1]+'.avi'])
            subprocess.call(['rm', '-r', out_path])
        if interact:
            window.show(ren)

    def save_mips(self, filename='spang_mips.pdf'):
        print('Writing '+filename)
        col_labels = np.apply_along_axis(util.j2str, 1, np.arange(self.J)[:,None])[None,:]
        viz.plot5d(filename, self.f[...,None], col_labels=col_labels)
            
    def save_tiff(self, filename='sh.tif', data=None):
        if data is None:
            data = self.f
        
        print('Writing '+filename)
        with tifffile.TiffWriter(filename, imagej=True) as tif:
            if data.ndim == 4:
                d = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
                tif.save(d[None,:,:,:,:]) # TZCYXS
            elif data.ndim == 3:
                d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
                tif.save(d[None,:,None,:,:].astype(np.float32)) # TZCYXS
                
    def read_tiff(self, filename):
        with tifffile.TiffFile(filename) as tf:
            self.f = np.moveaxis(tf.asarray(), [0, 1, 2, 3], [2, 3, 1, 0])

    def save_stats(self, folder='./'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.save_tiff(filename=folder+'sh.tif', data=self.f)
        self.save_tiff(filename=folder+'density.tif', data=self.density())
        self.save_tiff(filename=folder+'gfa.tif', data=self.gfa())
        tensor, principal = self.tensor()
        self.save_tiff(filename=folder+'tensor.tif', data=tensor)
        self.save_tiff(filename=folder+'principal.tif', data=principal)
