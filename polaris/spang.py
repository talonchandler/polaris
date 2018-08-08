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
        self.X = f.shape[0]
        self.Y = f.shape[1]
        self.Z = f.shape[2]
        
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
        
    def calc_B(self):
        # Calculate odf to sh matrix
        B = np.zeros((len(self.sphere.theta), self.f.shape[-1]))
        for index, x in np.ndenumerate(B):
            l, m = util.j2lm(index[1])
            B[index] = util.spZnm(l, m, self.sphere.theta[index[0]], self.sphere.phi[index[0]])
        self.B = B
        self.Binv = np.linalg.pinv(self.B, rcond=1e-15)

    def calc_stats(self, mask=None):
        print('Calculating summary statistics')
        density = np.zeros((self.X, self.Y, self.Z))
        self.gfa = np.zeros((self.X, self.Y, self.Z))
        self.peak_dirs = np.zeros((self.X, self.Y, self.Z, 3))
        self.peak_values = np.zeros((self.X, self.Y, self.Z))

        if mask is None:
            mask = np.ones((self.X, self.Y, self.Z))
        
        for x in tqdm(range(self.X)):
            for y in range(self.Y):
                for z in range(self.Z):
                    index = (x,y,z)                    
                    if mask[index]:
                        # Calculate spherical Fourier transform
                        odf = np.matmul(self.Binv.T, self.f[index]).clip(min=0)

                        # Calculate density
                        density[index] = np.sum(odf)
                        self.peak_dirs[index] = self.sphere.vertices[np.argmax(odf)]
                        self.peak_values[index] = np.amax(odf)

                        # Calculate gfa
                        std = np.std(odf, axis=-1)
                        rms = np.sqrt(np.mean(odf**2, axis=-1))
                        self.gfa[index] = np.divide(std, rms, where=(rms != 0))

        self.density = density/np.max(density)
        self.maxpeak = np.max(self.peak_values)

    def save_mips(self, filename='spang_mips.pdf'):
        col_labels = np.apply_along_axis(util.j2str, 1, np.arange(self.S)[:,None])[None,:]
        viz.plot5d(filename, self.f[:,:,:,:,None], col_labels=col_labels)

    def save_fft(self, filename='fft.pdf'):
        axes = (0,1,2)
        myfft = np.abs(np.fft.fftn(self.f, axes=axes, norm='ortho'))
        ffts = np.fft.fftshift(myfft, axes=axes)
        col_labels = np.apply_along_axis(util.j2str, 1, np.arange(self.S)[:,None])[None,:]
        viz.plot5d(filename, ffts[:,:,:,:,None], col_labels=col_labels)

    def ppos(self):
        # Project onto positive values
        # TODO is this right? Binv swapped with B?
        odf = np.einsum('ijkl,ml->ijkm', self.f, self.B)
        odf = odf.clip(min=0)
        self.f = np.einsum('ijkl,ml->ijkm', odf, self.Binv)

    def save_summary(self, filename='out.pdf', gfa_filter=None, mag=4,
                     mask=None, scale=1):
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
                                   outer_box=False, axes=False,
                                   clip_neg=False, azimuth=0, elevation=0,
                                   n_frames=1, mag=mag, video=False, scale=scale,
                                   interact=False, viz_type=viz_type,
                                   save_parallels=True, mask=mask)
                    
                    viz.plot_images(['parallels/yz.png', 'parallels/xy.png', 'parallels/xz.png'],
                                    f, spec, row, col,
                                    col_labels=col_labels, row_labels=None,
                                    vmin=vmin, vmax=vmax, colormap=colormap,
                                    cols=cols, yscale_label=None, pos=pos)
                    subprocess.call(['rm', '-r', 'parallels'])
                    
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
        
    def visualize(self, out_path='out/', outer_box=True, axes=True,
                  clip_neg=False, azimuth=0, elevation=0, n_frames=1,
                  size=(600,600), mag=4, video=True, viz_type='ODF', mask=None,
                  scale=1, zoom=1.0, zoom_in=1.0, interact=False, save_parallels=False):
        print('Preparing to render ' + out_path)
        
        # Prepare output
        import os
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        # Mask
        if mask is None:
            mask = np.ones((self.X, self.Y, self.Z), dtype=np.bool)
        for x in [-1,0]:
            for y in [-1,0]:
                for z in [-1,0]:
                    mask[x,y,z] = True

        # Render
        ren = window.Renderer()
        ren.background([1,1,1])

        # Add visuals to renderer
        if viz_type == "ODF":
            fodf_spheres = viz.odf_sparse(self.f, self.Binv, self.maxpeak,
                                          sphere=self.sphere, scale=scale*0.5/self.maxpeak,
                                          norm=False, colormap='bwr', mask=mask,
                                          global_cm=True)
            ren.add(fodf_spheres)
        elif viz_type == "PEAK":
            fodf_peaks = viz.peak_slicer(self.peak_dirs[:,:,:,None,:],
                                         self.peak_values[:,:,:,None]*scale*0.5/self.maxpeak, mask=mask)
            ren.add(fodf_peaks)

        X = self.X - 1
        Y = self.Y - 1
        Z = self.Z - 1

        if outer_box:
            c = np.array([0,0,0])
            ren.add(actor.line([np.array([[0,0,0],[X,0,0],[X,Y,0],[0,Y,0],
                                          [0,0,0],[0,Y,0],[0,Y,Z],[0,0,Z],
                                          [0,0,0],[X,0,0],[X,0,Z],[0,0,Z]])], colors=c))
            ren.add(actor.line([np.array([[X,0,Z],[X,Y,Z],[X,Y,0],[X,Y,Z],
                                          [0,Y,Z]])], colors=c))
        if axes:
            NN = np.max([X, Y, Z])
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
        if interact:
            window.show(ren)

    def save_tiff(self, filename):
        # Writes each spherical harmonic to a tiff. Not in use. 
        print('Writing '+filename)
        import tifffile
        for sh in range(self.f.shape[-1]):
            with tifffile.TiffWriter(filename+str(sh)+'.tiff', bigtiff=True) as tif:
                tif.save(self.f[...,sh])
                
            
