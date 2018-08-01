import subprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from polaris import viz, util
import numpy as np
from dipy.viz import window, actor
import vtk
from tqdm import tqdm

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
        self.calc_B()

    def save_tiff(self, filename):
        # Writes each spherical harmonic to a tiff. Not in use. 
        print('Writing '+filename)
        import tifffile
        for sh in range(self.f.shape[-1]):
            with tifffile.TiffWriter(filename+str(sh)+'.tiff', bigtiff=True) as tif:
                tif.save(self.f[...,sh])

    def calc_stats(self):
        # Calculate spherical Fourier transform
        # self.odf = np.einsum('ijkl,ml->ijkm', self.f, self.B)
        odf = np.einsum('ijkl,ml->ijkm', self.f, self.B).clip(min=0)
        self.odf = odf/np.max(odf) # clipped and normalized
        density = np.sum(self.odf, axis=-1)
        self.density = density/np.max(density)
        self.std = np.std(self.odf, axis=-1)
        self.rms = np.sqrt(np.mean(self.odf**2, axis=-1))
        gfa = np.where(self.rms > 1e-15, self.std/self.rms, 0)
        self.gfa = np.where(self.density > 0.1, gfa, 0)

    def save_mips(self, filename='spang_mips.pdf'):
        col_labels = np.apply_along_axis(util.j2str, 1, np.arange(self.S)[:,None])[None,:]
        viz.plot5d(filename, self.f[:,:,:,:,None], col_labels=col_labels)

    def save_fft(self, filename='fft.pdf'):
        axes = (0,1,2)
        myfft = np.abs(np.fft.fftn(self.f, axes=axes, norm='ortho'))
        ffts = np.fft.fftshift(myfft, axes=axes)
        col_labels = np.apply_along_axis(util.j2str, 1, np.arange(self.S)[:,None])[None,:]
        viz.plot5d(filename, ffts[:,:,:,:,None], col_labels=col_labels)

    def calc_B(self):
        # Calculate odf to sh matrix
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')
        B = np.zeros((len(sphere.theta), self.f.shape[-1]))
        for index, x in np.ndenumerate(B):
            l, m = util.j2lm(index[1])
            B[index] = util.spZnm(l, m, sphere.theta[index[0]], sphere.phi[index[0]])
        self.B = B
        self.Binv = np.linalg.pinv(self.B, rcond=1e-15)

    def ppos(self):
        # Project onto positive values
        odf = np.einsum('ijkl,ml->ijkm', self.f, self.B)
        odf = odf.clip(min=0)
        self.f = np.einsum('ijkl,ml->ijkm', odf, self.Binv)
        
    def visualize(self, out_path='out/', zoom=1.0, outer_box=True, axes=True,
                  clip_neg=False, azimuth=0, azimuth2=0, elevation=0, parallel=False,
                  scale=0.5, n_frames=1, size=(600, 600), mag=4, video=True,
                  odfs=True, interact=False):

        # Prepare output
        import os
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Calculate spherical Fourier transform
        odf = np.einsum('ijkl,ml->ijkm', self.f, self.B)
        if clip_neg:
            odf = odf.clip(min=0)

        # print('ODF Min: ', np.min(odf), '\t ODF Max: ', np.max(odf))
        
        # Render
        ren = window.Renderer()
        ren.background([1,1,1])

        # kludges for correct color scales
        if np.max(odf) > np.abs(np.min(odf)):
            odf = odf/np.max(odf)
        else:
            odf = odf/np.abs(np.min(odf))
        odf[0,0,0,0] = -1

        # Add visuals to renderer
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')
        if odfs:
            fodf_spheres = viz.odf_slicer(odf, sphere=sphere, scale=scale,
                                          norm=False, colormap='bwr', mask=None,
                                          global_cm=True)
            ren.add(fodf_spheres)
        else:
            max_ind = np.argmax(odf, axis=-1)
            xx = sphere.x[max_ind]
            yy = sphere.y[max_ind]
            zz = sphere.z[max_ind]
            dirs = np.stack([xx, yy, zz], axis=-1)
            peak_values = 0.5*np.amax(odf, axis=-1)
            fodf_peaks = viz.peak_slicer(dirs[:,:,:,None,:], peak_values[:,:,:,None])
            ren.add(fodf_peaks)

        NX = self.NX - 1
        NY = self.NY - 1
        NZ = self.NZ - 1

        if outer_box:
            c = np.array([0,0,0])
            ren.add(actor.line([np.array([[0,0,0],[NX,0,0],[NX,NY,0],[0,NY,0],
                                          [0,0,0],[0,NY,0],[0,NY,NZ],[0,0,NZ],
                                          [0,0,0],[NX,0,0],[NX,0,NZ],[0,0,NZ]])], colors=c))
            ren.add(actor.line([np.array([[NX,0,NZ],[NX,NY,NZ],[NX,NY,0],[NX,NY,NZ],
                                          [0,NY,NZ]])], colors=c))
        if axes:
            NN = np.max([NX, NY, NZ])
            ren.add(actor.line([np.array([[0,0,0],[NN/10,0,0]])], colors=np.array([1,0,0]), linewidth=4))
            ren.add(actor.line([np.array([[0,0,0],[0,NN/10,0]])], colors=np.array([0,1,0]), linewidth=4))
            ren.add(actor.line([np.array([[0,0,0],[0,0,NN/10]])], colors=np.array([0,0,1]), linewidth=4))

        # Setup vtk renderers
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(size[0], size[1])
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        ren.ResetCamera()
        ren.azimuth(azimuth)
        ren.elevation(elevation)
        ren.roll(azimuth2)
        if parallel:
            ren.projection(proj_type='parallel')
        ren.zoom(zoom)

        writer = vtk.vtkPNGWriter()
        az = 0
        naz = np.ceil(360/n_frames)
        
        print('Rendering ' + out_path)
        for i in tqdm(range(n_frames)):
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
        fps = np.ceil(n_frames/12)
        if video:
            subprocess.call(['ffmpeg', '-nostdin', '-y', '-framerate', str(fps),
                             '-loglevel', 'panic',
                             '-i', out_path+'%06d.png', out_path[:-1]+'.mp4'])
        
        if interact:
            window.show(ren)

    def save_summary(self, filename='out.pdf'):
        print('Generating ' + filename)
        self.calc_stats()
        pos = (-0.05, 1.05, 0.5, 0.55) # Arrow and label positions
        vmin = 0
        vmax = 1
        colormap = 'gray'
        inches = 2
        rows = 1
        cols = 5
        widths = [1.1]*(cols - 1) + [0.05]
        heights = [1]*rows
        col_labels = np.array([['Spatio-angular density', 'Peak', 'Density', 'GFA[Density $>$ 0.2]']])
        f = plt.figure(figsize=(inches*np.sum(widths), inches*np.sum(heights)))
        spec = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths,
                                 height_ratios=heights, hspace=0.25, wspace=0.15)
        for row in range(rows):
            for col in range(cols):
                if col == 0 or col == 1:
                    if col == 1:
                        odfs = False
                    else:
                        odfs = True
                    
                    self.visualize(out_path='yz/', zoom=1.7, outer_box=False, axes=False,
                                   clip_neg=True, azimuth=90, azimuth2=0, elevation=0, parallel=True,
                                   scale=0.5, n_frames=1, mag=4, video=False,
                                   interact=False, odfs=odfs)
                    self.visualize(out_path='xy/', zoom=1.7, outer_box=False, axes=False,
                                   clip_neg=True, azimuth=0, elevation=0, parallel=True,
                                   scale=0.5, n_frames=1, mag=4, video=False,
                                   interact=False, odfs=odfs)
                    self.visualize(out_path='xz/', zoom=1.7, outer_box=False, axes=False,
                                   clip_neg=True, azimuth=0, elevation=90, parallel=True,
                                   scale=0.5, n_frames=1, mag=4, video=False,
                                   interact=False, odfs=odfs)

                    viz.plot_images(['yz/000000.png', 'xy/000000.png', 'xz/000000.png'],
                                    f, spec, row, col,
                                    col_labels=col_labels, row_labels=None,
                                    vmin=vmin, vmax=vmax, colormap=colormap,
                                    cols=cols, yscale_label=None, pos=pos)
                    subprocess.call(['rm', '-r', 'yz', 'xy', 'xz'])
                    
                elif col == 2:
                    viz.plot_projections(self.density,
                                         f, spec, row, col,
                                         col_labels=col_labels, row_labels=None,
                                         vmin=vmin, vmax=vmax, colormap=colormap,
                                         cols=cols, yscale_label=None, pos=pos)
                elif col == 3:
                    viz.plot_projections(self.gfa,
                                         f, spec, row, col,
                                         col_labels=col_labels, row_labels=None,
                                         vmin=vmin, vmax=vmax, colormap=colormap,
                                         cols=cols, yscale_label=None, pos=pos)
                elif col == 4:
                    viz.plot_colorbar(f, spec, row, col, vmin, vmax, colormap)

        f.savefig(filename, bbox_inches='tight')
