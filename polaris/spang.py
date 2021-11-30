import subprocess
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', family='serif', serif='CMU Serif')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
#rc('text', usetex=True)
from polaris import viz, util
import numpy as np
from dipy.viz import window, actor
from dipy.data import get_sphere
import vtk
from tqdm import tqdm
import tifffile
import os
import logging
log = logging.getLogger('log')

class Spang:
    """
    A Spang (short for spatio-angular density) is a representation of a 
    spatio-angular density f(r, s) stored as a 4D array of voxel values 
    and spherical harmonic coefficients [x, y, z, j]. A Spang object is 
    a discretized member of object space U. 
    """
    def __init__(self, f=np.zeros((3,3,3,15), dtype=np.float32),
                 vox_dim=(1,1,1), sphere=get_sphere('symmetric724')):
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
        self.sphere = sphere.subdivide()

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

    def density(self, norm=True):
        if norm:
            return self.f[...,0]/np.max(self.f[...,0])
        else:
            return self.f[...,0]
        
    def gfa(self):
        return np.nan_to_num(np.sqrt(1 - (self.f[...,0]**2)/np.sum(self.f**2, axis=-1)))

    def tensor(self):
        log.info("Calculating tensor fits")
        M = np.load(os.path.join(os.path.dirname(__file__), 'harmonics/sh2tensor.npy'))
        Di = np.einsum('ijkl,lm->ijkm', self.f[...,0:6], M)
        D = np.zeros(self.f.shape[0:3]+(3,3), dtype=np.float32)
        D[...,0,0] = Di[...,0]; D[...,0,1] = Di[...,3]; D[...,0,2] = Di[...,5];
        D[...,1,0] = Di[...,3]; D[...,1,1] = Di[...,1]; D[...,1,2] = Di[...,4];
        D[...,2,0] = Di[...,5]; D[...,2,1] = Di[...,4]; D[...,2,2] = Di[...,2];
        eigs = np.linalg.eigh(D)
        principal = eigs[1][...,-1]*eigs[1][...,-1]
        return Di.astype(np.float32), principal.astype(np.float32)
        
    def save_summary(self, filename='out.pdf', density_filter=None, mag=4,
                     mask=None, scale=1.0, keep_parallels=False, skip_n=1):
        log.info('Generating ' + filename)
        if density_filter is not None:
            density_mask = self.density() > density_filter
            mask = np.logical_or(mask, density_mask).astype(np.bool)
        pos = (-0.05, 1.05, 0.5, 0.55) # Arrow and label positions
        vmin = 0
        vmax = 1
        inches = 4
        rows = 2
        cols = 3
        colormap = 'Reds'
        widths = [1]*cols
        heights = [1]*rows
        M = np.max(self.f.shape)
        x_frac = self.f.shape[0]/M
        if density_filter is None:
            filter_label = ''
        else:
            filter_label = '\n where density $>$ ' + str(density_filter)
        if skip_n == 1:
            skip_label = ''
        else:
            skip_label = '\n downsampled ' + str(skip_n) + '$\\times$'
        col_labels = np.array([['ODF', 'Density', 'GFA'], ['Peak', 'Ellipsoid', 'Principal']])
        f = plt.figure(figsize=(inches*np.sum(widths), inches*np.sum(heights)))
        spec = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths,
                                 height_ratios=heights, hspace=0.1, wspace=0.075)
        for row in range(rows):
            for col in range(cols):
                if col < 3:
                    yscale_label = None
                    if row == 0 and col == 0:
                        bar = True
                        bar_label = 'ODF radius' + skip_label + filter_label
                        colormap = 'Reds'
                        self.visualize(out_path='parallels/', zoom_start=1.7,
                                       outer_box=False, axes=False,
                                       clip_neg=False, azimuth=0, elevation=0,
                                       n_frames=1, mag=mag, video=False, scale=scale,
                                       interact=False, viz_type='ODF',
                                       save_parallels=True, mask=mask, skip_n=skip_n)
                    if row == 1 and col == 1:
                        bar = False
                        bar_label = 'Principal' + skip_label + filter_label
                        self.visualize(out_path='parallels/', zoom_start=1.7,
                                       outer_box=False, axes=False,
                                       clip_neg=False, azimuth=0, elevation=0,
                                       n_frames=1, mag=mag, video=False, scale=scale,
                                       interact=False, viz_type='Ellipsoid',
                                       save_parallels=True, mask=mask, skip_n=skip_n)
                    if row == 1 and col == 2:
                        bar = False
                        bar_label = 'Principal' + skip_label + filter_label
                        self.yscale = 1e-3*self.vox_dim[1]*self.f.shape[0]
                        yscale_label = '{:.2f}'.format(self.yscale) + ' $\mu$m'
                        self.visualize(out_path='parallels/', zoom_start=1.7,
                                       outer_box=False, axes=False,
                                       clip_neg=False, azimuth=0, elevation=0,
                                       n_frames=1, mag=mag, video=False, scale=scale,
                                       interact=False, viz_type='Principal',
                                       save_parallels=True, mask=mask, skip_n=skip_n)
                    if row == 1 and col == 0:
                        bar = False
                        bar_label = 'Peak' + skip_label + filter_label
                        self.visualize(out_path='parallels/', zoom_start=1.7,
                                       outer_box=False, axes=False,
                                       clip_neg=False, azimuth=0, elevation=0,
                                       n_frames=1, mag=mag, video=False, scale=scale,
                                       interact=False, viz_type='Peak',
                                       save_parallels=True, mask=mask, skip_n=skip_n)
                    if row == 0 and col == 1:
                        colormap = 'gray'
                        bar = True
                        bar_label = 'Density'
                        viz.plot_parallels(self.density(), out_path='parallels/', outer_box=False,
                                           axes=False, clip_neg=False, azimuth=0,
                                           elevation=0, scale=scale)
                    if row == 0 and col == 2:
                        colormap = 'gray'
                        bar = True
                        bar_label = 'GFA' + filter_label
                        viz.plot_parallels(self.gfa(), out_path='parallels/', outer_box=False,
                                           axes=False, clip_neg=False, azimuth=0,
                                           elevation=0, scale=scale, mask=mask)

                    viz.plot_images(['parallels/yz.tif', 'parallels/xy.tif', 'parallels/xz.tif'],
                                    f, spec, row, col,
                                    col_labels=col_labels, row_labels=None,
                                    vmin=vmin, vmax=vmax, colormap=colormap,
                                    rows=rows, cols=cols, x_frac=x_frac,
                                    yscale_label=yscale_label, pos=pos, bar=bar, bar_label=bar_label)
                    #if not keep_parallels:
                        #subprocess.call(['rm', '-r', 'parallels'])
                    
                elif col == 3:
                    viz.plot_colorbar(f, spec, row, col, vmin, vmax, colormap)

        log.info('Saving ' + filename)
        f.savefig(filename, bbox_inches='tight')

    def save_mips(self, filename='spang_mips.pdf'):
        log.info('Writing '+filename)
        col_labels = np.apply_along_axis(util.j2str, 1, np.arange(self.J)[:,None])[None,:]
        viz.plot5d(filename, self.f[...,None], col_labels=col_labels)
            
    def save_tiff(self, filename='sh.tif', data=None):
        util.mkdir(filename)
        if data is None:
            data = self.f
        
        log.info('Writing '+filename)
        with tifffile.TiffWriter(filename, imagej=True) as tif:
            if data.ndim == 4:
                dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
                tif.save(dat[None,:,:,:,:].astype(np.float32)) # TZCYXS
            elif data.ndim == 3:
                d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
                tif.save(d[None,:,None,:,:].astype(np.float32)) # TZCYXS
                
    def read_tiff(self, filename):
        log.info('Reading '+filename)
        with tifffile.TiffFile(filename) as tf:
            self.f = np.ascontiguousarray(np.moveaxis(tf.asarray(), [0, 1, 2, 3], [2, 3, 1, 0]))
        self.X = self.f.shape[0]
        self.Y = self.f.shape[1]
        self.Z = self.f.shape[2]

    def save_stats(self, folder='./', save_sh=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        if save_sh:
            self.save_tiff(filename=folder+'sh.tif', data=self.f)
        self.save_tiff(filename=folder+'density.tif', data=self.density())
        self.save_tiff(filename=folder+'gfa.tif', data=self.gfa())
        
    def visualize(self, out_path='out/', outer_box=True, axes=True,
                  clip_neg=False, azimuth=0, elevation=0, n_frames=1, mag=1,
                  video=False, viz_type='ODF', mask=None, mask_roi=None,
                  skip_n=1, skip_n_roi=1, scale=1, roi_scale=1, zoom_start=1.0,
                  zoom_end=1.0, top_zoom=1, interact=False,
                  save_parallels=False, my_cam=None, compress=True, roi=None,
                  corner_text='', scalemap=None, titles_on=True,
                  scalebar_on=True, invert=False, flat=False, colormap='bwr',
                  global_cm=True, camtilt=False, axes_on=False, colors=None,
                  arrows=None, arrow_color=np.array([0,0,0]), linewidth=0.1,
                  mark_slices=None, z_shift=0, profiles=[], markers=[],
                  marker_colors=[], marker_scale=1, normalize_glyphs=True,
                  gamma=1, density_max=1):
        log.info('Preparing to render ' + out_path)

        # Handle scalemap
        if scalemap is None:
            scalemap = util.ScaleMap(min=np.min(self.f[...,0]), max=np.max(self.f[...,0]))
        
        # Prepare output
        util.mkdir(out_path)
            
        # Setup vtk renderers
        renWin = vtk.vtkRenderWindow()
        
        if not interact:
            renWin.SetOffScreenRendering(1)
        if isinstance(viz_type, str):
            viz_type = [viz_type]
            
        # Rows and columns
        cols = len(viz_type)
        if roi is None:
            rows = 1
        else:
            rows = 2
            
        renWin.SetSize(np.int(500*mag*cols), np.int(500*mag*rows))

        # Select background color
        if save_parallels:
            bg_color = [1,1,1]
            line_color = np.array([0,0,0])
            line_bcolor = np.array([1,1,1])
        else:
            if not invert:
                bg_color = [0,0,0]
                line_color = np.array([1,1,1])
                line_bcolor = np.array([0,0,0])
            else:
                bg_color = [1,1,1]
                line_color = np.array([0,0,0])
                line_bcolor = np.array([1,1,1])

        # For each viz_type
        rens = []
        zoom_start = []
        zoom_end = []
        for row in range(rows):
            for col in range(cols):
                # Render
                ren = window.Scene()
                rens.append(ren)
                if viz_type[col] is 'Density':
                    ren.background([0,0,0])
                    line_color = np.array([1,1,1])                    
                else:
                    ren.background(bg_color)
                ren.SetViewport(col/cols,(rows - row - 1)/rows,(col+1)/cols,(rows - row)/rows)
                renWin.AddRenderer(ren)
                iren = vtk.vtkRenderWindowInteractor()
                iren.SetRenderWindow(renWin)

                # Mask
                if mask is None:
                    mask = np.ones((self.X, self.Y, self.Z), dtype=np.bool)
                if mask_roi is None:
                    mask_roi = mask

                # Main vs roi
                if row == 0:
                    data = self.f
                    skip_mask = np.zeros(mask.shape, dtype=np.bool)
                    skip_mask[::skip_n,::skip_n,::skip_n] = 1
                    my_mask = np.logical_and(mask, skip_mask)
                    scale = scale
                    scalemap = scalemap
                    if np.sum(my_mask) == 0:
                        my_mask[0,0,0] = True
                else:
                    data = self.f[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1], roi[0][2]:roi[1][2], :]
                    roi_mask = mask_roi[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1], roi[0][2]:roi[1][2]]
                    skip_mask = np.zeros(roi_mask.shape, dtype=np.bool)
                    skip_mask[::skip_n_roi,::skip_n_roi,::skip_n_roi] = 1
                    my_mask = np.logical_and(roi_mask, skip_mask)
                    scale = roi_scale
                    scalemap = scalemap

                # Add visuals to renderer
                if viz_type[col] == "ODF":
                    renWin.SetMultiSamples(4) 
                    log.info('Rendering '+str(np.sum(my_mask)) + ' ODFs')
                    fodf_spheres = viz.odf_sparse(data, self.Binv, sphere=self.sphere,
                                                  scale=skip_n*scale*0.5, norm=False,
                                                  colormap=colormap, mask=my_mask,
                                                  global_cm=global_cm, scalemap=scalemap,
                                                  odf_sphere=False, flat=flat, normalize=normalize_glyphs)

                    ren.add(fodf_spheres)
                elif viz_type[col] == "ODF Sphere":
                    renWin.SetMultiSamples(4)                     
                    log.info('Rendering '+str(np.sum(my_mask)) + ' ODFs')
                    fodf_spheres = viz.odf_sparse(data, self.Binv, sphere=self.sphere,
                                                  scale=skip_n*scale*0.5, norm=False,
                                                  colormap=colormap, mask=my_mask,
                                                  global_cm=global_cm, scalemap=scalemap,
                                                  odf_sphere=True, flat=flat)
                    ren.add(fodf_spheres)
                elif viz_type[col] == "Ellipsoid":
                    renWin.SetMultiSamples(4)                     
                    log.info('Warning: scaling is not implemented for ellipsoids')                    
                    log.info('Rendering '+str(np.sum(my_mask)) + ' ellipsoids')
                    fodf_peaks = viz.tensor_slicer_sparse(data,
                                                          sphere=self.sphere,
                                                          scale=skip_n*scale*0.5,
                                                          mask=my_mask)
                    ren.add(fodf_peaks)
                elif viz_type[col] == "Peak":
                    renWin.SetMultiSamples(4)                     
                    log.info('Rendering '+str(np.sum(my_mask)) + ' peaks')
                    fodf_peaks = viz.peak_slicer_sparse(data, self.Binv, self.sphere.vertices, 
                                                        linewidth=linewidth, scale=skip_n*scale*0.5, colors=colors,
                                                        mask=my_mask, scalemap=scalemap, normalize=normalize_glyphs)
                    fodf_peaks.GetProperty().LightingOn()
                    fodf_peaks.GetProperty().SetDiffuse(0.4) # Doesn't work (VTK bug I think)
                    fodf_peaks.GetProperty().SetAmbient(0.15)
                    fodf_peaks.GetProperty().SetSpecular(0)
                    fodf_peaks.GetProperty().SetSpecularPower(0)

                    ren.add(fodf_peaks)
                elif viz_type[col] == "Principal":
                    log.info('Warning: scaling is not implemented for principals')
                    log.info('Rendering '+str(np.sum(my_mask)) + ' principals')
                    fodf_peaks = viz.principal_slicer_sparse(data, self.Binv, self.sphere.vertices,
                                                             scale=skip_n*scale*0.5,
                                                             mask=my_mask)
                    ren.add(fodf_peaks)
                elif viz_type[col] == "Density":
                    renWin.SetMultiSamples(0) # Must be zero for smooth
                    renWin.SetAAFrames(4) # Slow antialiasing for volume renders
                    log.info('Rendering density')
                    gamma_corr = np.where(data[...,0]>0, data[...,0]**gamma, data[...,0])
                    scalemap.max = density_max*scalemap.max**gamma
                    volume = viz.density_slicer(gamma_corr, scalemap)
                    ren.add(volume)

                X = np.float(data.shape[0])
                Y = np.float(data.shape[1])
                Z = np.float(data.shape[2]) - z_shift
                
                # Titles                
                if row == 0 and titles_on:
                    viz.add_text(ren, viz_type[col], 0.5, 0.96, mag)
                    
                # Scale bar
                if col == cols - 1 and not save_parallels and scalebar_on:
                    yscale = 1e-3*self.vox_dim[1]*data.shape[1]
                    yscale_label = '{:.2g}'.format(yscale) + ' um'
                    viz.add_text(ren, yscale_label, 0.5, 0.03, mag)
                    viz.draw_scale_bar(ren, X, Y, Z, [1,1,1])

                # Corner text
                if row == rows - 1 and col == 0 and titles_on:
                    viz.add_text(ren, corner_text, 0.03, 0.03, mag, ha='left')

                # Draw boxes
                Nmax = np.max([X, Y, Z])
                if outer_box:
                    if row == 0:
                        viz.draw_outer_box(ren, np.array([[0,0,0],[X,Y,Z]]) - 0.5, line_color)
                    if row == 1:
                        viz.draw_outer_box(ren, np.array([[0,0,0],[X,Y,Z]]) - 0.5, [0,1,1])

                # Add colored axes
                if axes:
                    viz.draw_axes(ren, np.array([[0,0,0], [X,Y,Z]]) - 0.5)

                # Add custom arrows
                if arrows is not None:
                    for i in range(arrows.shape[0]):
                        viz.draw_single_arrow(ren, arrows[i,0,:], arrows[i,1,:], color=arrow_color)
                        viz.draw_unlit_line(ren, [np.array([arrows[i,0,:],[X/2,Y/2,Z/2]])], [arrow_color], lw=0.3, scale=1.0)

                # Draw roi box
                if row == 0 and roi is not None:
                    maxROI = np.max([roi[1][0] - roi[0][0], roi[1][1] - roi[0][1], roi[1][2] - roi[0][2]])
                    maxXYZ = np.max([self.X, self.Y, self.Z])
                    viz.draw_outer_box(ren, roi, [0,1,1], lw=0.3*maxXYZ/maxROI)
                    viz.draw_axes(ren, roi, lw=0.3*maxXYZ/maxROI)
                    

                # Draw marked slices
                if mark_slices is not None:
                    for slicen in mark_slices:
                        md = np.max((X, Z))
                        frac = slicen/data.shape[1]
                        rr = 0.83*md
                        t1 = 0
                        t2 = np.pi/2 
                        t3 = np.pi
                        t4 = 3*np.pi/2
                        points = [np.array([[X/2+rr*np.cos(t1),frac*Y,Z/2+rr*np.sin(t1)],
                                            [X/2+rr*np.cos(t2),frac*Y,Z/2+rr*np.sin(t2)],
                                            [X/2+rr*np.cos(t3),frac*Y,Z/2+rr*np.sin(t3)],
                                            [X/2+rr*np.cos(t4),frac*Y,Z/2+rr*np.sin(t4)],
                                            [X/2+rr*np.cos(t1),frac*Y,Z/2+rr*np.sin(t1)],
                                            [X/2+rr*np.cos(t2),frac*Y,Z/2+rr*np.sin(t2)]])]
                        viz.draw_unlit_line(ren, points, 6*[line_color+0.6], lw=0.3, scale=1.0)

                # Draw markers
                for i, marker in enumerate(markers):
                    # Draw sphere
                    source = vtk.vtkSphereSource()
                    source.SetCenter(marker)
                    source.SetRadius(marker_scale)
                    source.SetThetaResolution(30)
                    source.SetPhiResolution(30)

                    # mapper
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(source.GetOutputPort())

                    # actor
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(marker_colors[i,:])
                    actor.GetProperty().SetLighting(0)
                    ren.AddActor(actor)
                        
                # Draw profile lines
                colors = np.array([[1,0,0],[0,1,0],[0,0,1]])

                for i, profile in enumerate(profiles):
                    import pdb; pdb.set_trace() 
                    n_seg = profile.shape[0]
                    viz.draw_unlit_line(ren, [profile], n_seg*[colors[i,:]], lw=0.5, scale=1.0)

                    # Draw sphere
                    source = vtk.vtkSphereSource()
                    source.SetCenter(profile[0])
                    source.SetRadius(1)
                    source.SetThetaResolution(30)
                    source.SetPhiResolution(30)

                    # mapper
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(source.GetOutputPort())

                    # actor
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    # actor.GetProperty().SetColor(colors[i,:])
                    actor.GetProperty().SetLighting(0)

                    # assign actor to the renderer
                    ren.AddActor(actor)

                # Setup cameras
                Rmax = np.linalg.norm([Z/2, X/2, Y/2])
                Rcam_rad = Rmax/np.tan(np.pi/12)        
                Ntmax = np.max([X, Y])
                ZZ = Z
                if ZZ > Ntmax:
                    Rcam_edge = np.max([X/2, Y/2])
                else:
                    Rcam_edge = np.min([X/2, Y/2])
                Rcam = Rcam_edge + Rcam_rad
                if my_cam is None:
                    cam = ren.GetActiveCamera()
                    if camtilt:
                        cam.SetPosition(((X-1)/2, (Y-1)/2, (Z-1)/2 + Rcam))
                        cam.SetViewUp((-1, 0, 1))
                        if axes_on:
                            max_dim = np.max((X, Z))
                            viz.draw_unlit_line(ren, [np.array([[(X- max_dim)/2,Y/2,Z/2],[X/2,Y/2,+Z/2],[X/2,Y/2,(Z + max_dim)/2]])], 3*[line_color], lw=max_dim/250, scale=1.0)
                    else:
                        cam.SetPosition(((X-1)/2 + Rcam, (Y-1)/2, (Z-1)/2))
                        cam.SetViewUp((0, 0, 1))
                    cam.SetFocalPoint(((X-1)/2, (Y-1)/2, (Z-1)/2))
                    #ren.reset_camera()
                else:
                    ren.set_camera(*my_cam)
                ren.azimuth(azimuth)
                ren.elevation(elevation)

                # Set zooming
                if save_parallels:
                    zoom_start.append(1.7)
                    zoom_end.append(1.7)
                else:
                    if row == 0:
                        zoom_start.append(1.3*top_zoom)
                        zoom_end.append(1.3*top_zoom)
                    else:
                        zoom_start.append(1.3)
                        zoom_end.append(1.3)

        # Setup writer
        writer = vtk.vtkTIFFWriter()
        if not compress:
            writer.SetCompressionToNoCompression()

        # Execute renders
        az = 90
        naz = np.ceil(360/n_frames)
        log.info('Rendering ' + out_path)
        if save_parallels:
            # Parallel rendering for summaries
            filenames = ['yz', 'xy', 'xz']
            zooms = [zoom_start[0], 1.0, 1.0]
            azs = [90, -90, 0]
            els = [0, 0, 90]
            ren.projection(proj_type='parallel')
            ren.reset_camera()
            for i in tqdm(range(3)):
                ren.zoom(zooms[i])
                ren.azimuth(azs[i])
                ren.elevation(els[i])
                ren.reset_clipping_range()
                renderLarge = vtk.vtkRenderLargeImage()
                renderLarge.SetMagnification(1)
                renderLarge.SetInput(ren)
                renderLarge.Update()
                writer.SetInputConnection(renderLarge.GetOutputPort())
                writer.SetFileName(out_path + filenames[i] + '.tif')
                writer.Write()
        else:
            # Rendering for movies 
            for j, ren in enumerate(rens):
                ren.zoom(zoom_start[j])
            for i in tqdm(range(n_frames)):
                for j, ren in enumerate(rens):
                    ren.zoom(1 + ((zoom_end[j] - zoom_start[j])/n_frames))
                    ren.azimuth(az)
                    ren.reset_clipping_range()

                renderLarge = vtk.vtkRenderLargeImage()
                renderLarge.SetMagnification(1)
                renderLarge.SetInput(ren)
                renderLarge.Update()
                writer.SetInputConnection(renderLarge.GetOutputPort())
                if n_frames != 1:
                    writer.SetFileName(out_path + str(i).zfill(3) + '.tif')
                else:
                    writer.SetFileName(out_path + '.tif')
                writer.Write()
                az = naz

        # Interactive
        if interact:
            window.show(ren)

        # Generate video (requires ffmpeg)
        if video:
            log.info('Generating video from frames')
            fps = np.ceil(n_frames/12)
            subprocess.call(['ffmpeg', '-nostdin', '-y', '-framerate', str(fps),
                             '-loglevel', 'panic', '-i', out_path+'%03d'+'.png',
                             '-pix_fmt', 'yuvj420p', '-vcodec', 'mjpeg',
                             out_path[:-1]+'.avi'])
            # subprocess.call(['rm', '-r', out_path])

        return my_cam

    def vis_profiles(self, filename, profilesi, dx=0.13):
        from scipy.interpolate import interpn

        out = []
        for profilei in profilesi:
            grid = (np.arange(self.X), np.arange(self.Y), np.arange(self.Z))
            out.append(interpn(grid, self.f[...,0], profilei))

        # Normalize            
        out = np.array(out)
        out = out/np.max(out)

        f, ax = plt.subplots(1, 1, figsize=(1.5,1.5))
        ax.set_xlabel('Position along profile ($\mu$m)')
        ax.set_ylabel('Density (AU)')
        ax.set_ylim([0,1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        for i, profilei in enumerate(profilesi):
            # Calculate x positions
            xpos = np.zeros(out.shape[-1])
            xpos[1:] = np.linalg.norm(profilei[1:,:] - profilei[0:-1,:], axis=-1)
            xpos = np.cumsum(xpos)*dx
            
            c = np.zeros(3)
            c[i] = 1
            ax.plot(xpos, out[i,:], '-', c=c, clip_on=False)
            ax.plot(0, out[i,0], 'o', c=c, ms=5-.5*i, clip_on=False)
        plt.savefig(filename, bbox_inches='tight')

class SpangSeries:
    """
    A SpangSeries is a class for handling and visualizing a series of Spang
    objects without loading all of the data into memory. The main attribute is 
    a list of file names.
    """
    def __init__(self, filenames, label, vox_dim=(1,1,1)):
        if not isinstance(filenames, list):#os.path.isdir(filenames):
            # If folder
            import glob
            self.filenames = sorted(glob.glob(filenames+'*.tif'))
        else:
            self.filenames = filenames
        self.label = label
        self.vox_dim = vox_dim

    def visualize(self, output, density_filter=0.1, density_filter_roi=0.1, mag=1, n_frames=1, hyperstack=None, **kwargs):
        util.mkdir(output)

        # Generate visuals frame by frame
        log.info('Generating frames')
        for i, filename in enumerate(self.filenames):
            sp = Spang(vox_dim=self.vox_dim)
            sp.read_tiff(filename)
            mask = sp.density() > density_filter
            mask_roi = sp.density() > density_filter_roi
            sp.visualize(output+str(i).zfill(3)+'/', mask=mask, mask_roi=mask_roi, corner_text=self.label(i),
                         n_frames=n_frames, mag=mag, **kwargs)
        
        # Generate uncompressed hyperstack from compressed files
        if hyperstack is not None:
            # Get image dimension of first file
            with tifffile.TiffFile(output+'000/000.tif') as tf:
                tiffshape = tf.asarray().shape

            # Read all frames into memory
            log.info('Reading frames into memory')                    
            data = np.zeros(tiffshape + (len(self.filenames),) + (n_frames,), dtype=np.uint8)
            for i in range(len(self.filenames)):        
                for j in range(n_frames):
                    filename = output+str(i).zfill(3)+'/'+str(j).zfill(3)+'.tif'
                    with tifffile.TiffFile(filename) as tf:
                        data[...,i,j] = tf.asarray()

            # Write to a single hyperstack
            log.info('Writing hyperstack')                    
            with tifffile.TiffWriter(hyperstack, imagej=True) as tif:
                d = np.moveaxis(data, [-1, -2], [1, 0])
                tif.save(data=d[:,:,None,:,:,:]) # TZCYXS
