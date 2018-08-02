import subprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from polaris import viz, util
import numpy as np
from dipy.viz import window, actor
from dipy.data import get_sphere
import vtk
from tqdm import tqdm

class Spang:
    """
    A Spang (short for spatio-angular density) is a representation of a 
    spatio-angular density f(r, s) stored as a 4D array of voxel values 
    and spherical harmonic coefficients [x, y, z, j]. A Spang object is 
    a discretized member of object space U. 
    """
    def __init__(self, f=np.zeros((3,3,3,1)), vox_dim=(1,1,1),
                 sphere=get_sphere('symmetric724')):
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
        self.sphere = sphere
        self.calc_B()

    def save_tiff(self, filename):
        # Writes each spherical harmonic to a tiff. Not in use. 
        print('Writing '+filename)
        import tifffile
        for sh in range(self.f.shape[-1]):
            with tifffile.TiffWriter(filename+str(sh)+'.tiff', bigtiff=True) as tif:
                tif.save(self.f[...,sh])

    def calc_stats(self):
        print('Calculating summary statistics')
        # Calculate spherical Fourier transform
        # self.odf = np.einsum('ijkl,ml->ijkm', self.f, self.B)
        odf = np.einsum('ijkl,ml->ijkm', self.f, self.B).clip(min=0)
        self.odf = odf/np.max(odf) # clipped and normalized
        density = np.sum(self.odf, axis=-1)
        self.density = density/np.max(density)

        # Calculate gfa
        self.std = np.std(self.odf, axis=-1)
        self.rms = np.sqrt(np.mean(self.odf**2, axis=-1))
        self.gfa = np.zeros(self.std.shape)
        np.divide(self.std, self.rms, out=self.gfa, where=(self.rms != 0))

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
        B = np.zeros((len(self.sphere.theta), self.f.shape[-1]))
        for index, x in np.ndenumerate(B):
            l, m = util.j2lm(index[1])
            B[index] = util.spZnm(l, m, self.sphere.theta[index[0]], self.sphere.phi[index[0]])
        self.B = B
        self.Binv = np.linalg.pinv(self.B, rcond=1e-15)

    def ppos(self):
        # Project onto positive values
        odf = np.einsum('ijkl,ml->ijkm', self.f, self.B)
        odf = odf.clip(min=0)
        self.f = np.einsum('ijkl,ml->ijkm', odf, self.Binv)

    def visualize(self, out_path='out/', zoom=1.0, outer_box=True, axes=True,
                  clip_neg=False, azimuth=0, elevation=0,
                  scale=0.5, n_frames=1, size=(600, 600), mag=4, video=True,
                  viz_type='ODF', interact=False, save_parallels=False):

        # Prepare output
        import os
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Calculate spherical Fourier transform
        odf = np.einsum('ijkl,ml->ijkm', self.f, self.B)
        if clip_neg:
            odf = odf.clip(min=0)
            
        # Render
        ren = window.Renderer()
        ren.background([1,1,1])


        # kludges for correct color scales
        if np.max(odf) > np.abs(np.min(odf)):
            odf = odf/np.max(odf)
        else:
            odf = odf/np.abs(np.min(odf))
        odf[0,0,0,0] = -1

        # Mask
        mask = np.sum(odf, axis=-1) > 0
        mask[0,0,0] = True

        # Add visuals to renderer
        if viz_type == "ODF":
            fodf_spheres = viz.odf_slicer(odf, sphere=self.sphere, scale=scale,
                                          norm=False, colormap='bwr', mask=mask,
                                          global_cm=True)
            ren.add(fodf_spheres)
        elif viz_type == "PEAK":
            max_ind = np.argmax(odf, axis=-1)
            self.peak_dirs = self.sphere.vertices[max_ind]
            self.peak_values = 0.5*np.amax(odf, axis=-1)
            fodf_peaks = viz.peak_slicer(self.peak_dirs[:,:,:,None,:],
                                         self.peak_values[:,:,:,None], mask=mask)
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

    def save_summary(self, filename='out.pdf', gfa_filter=None, mag=4):
        self.calc_stats()
        print('Generating ' + filename)
        pos = (-0.05, 1.05, 0.5, 0.55) # Arrow and label positions
        vmin = 0
        vmax = 1
        colormap = 'gray'
        inches = 2
        rows = 1
        cols = 5
        widths = [1.1]*(cols - 1) + [0.05]
        heights = [1]*rows
        if gfa_filter is None:
            gfa_label = 'GFA'
        else:
            gfa_label = 'GFA[Density $>$ ' + str(gfa_filter)+']'
        col_labels = np.array([['Spatio-angular density', 'Peak', 'Density', gfa_label]])
        f = plt.figure(figsize=(inches*np.sum(widths), inches*np.sum(heights)))
        spec = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths,
                                 height_ratios=heights, hspace=0.25, wspace=0.15)
        for row in range(rows):
            for col in range(cols):
                if col == 0 or col == 1:
                    if col == 0:
                        viz_type = 'ODF'
                    else:
                        viz_type = 'PEAK'
                    
                    self.visualize(out_path='parallels/', zoom=1.7,
                                   outer_box=True, axes=False,
                                   clip_neg=True, azimuth=0, elevation=0,
                                   scale=0.5, n_frames=1, mag=mag, video=False,
                                   interact=False, viz_type=viz_type,
                                   save_parallels=True)

                    viz.plot_images(['parallels/yz.png', 'parallels/xy.png', 'parallels/xz.png'],
                                    f, spec, row, col,
                                    col_labels=col_labels, row_labels=None,
                                    vmin=vmin, vmax=vmax, colormap=colormap,
                                    cols=cols, yscale_label=None, pos=pos)
                    # subprocess.call(['rm', '-r', 'parallels'])
                    
                elif col == 2:
                    viz.plot_projections(self.density,
                                         f, spec, row, col,
                                         col_labels=col_labels, row_labels=None,
                                         vmin=vmin, vmax=vmax, colormap=colormap,
                                         cols=cols, yscale_label=None, pos=pos)
                elif col == 3:
                    if gfa_filter is None:
                        gfa = self.gfa
                    else:
                        gfa = self.gfa*(self.density > gfa_filter)
                    viz.plot_projections(gfa, f, spec, row, col,
                                         col_labels=col_labels, row_labels=None,
                                         vmin=vmin, vmax=vmax, colormap=colormap,
                                         cols=cols, yscale_label=None, pos=pos)
                elif col == 4:
                    viz.plot_colorbar(f, spec, row, col, vmin, vmax, colormap)

        print('Saving ' + filename)
        f.savefig(filename, bbox_inches='tight')
