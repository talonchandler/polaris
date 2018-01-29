import numpy as np
from polharmonic import util
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mayavi import mlab
import matplotlib.image as mpimg
import subprocess
import scipy.ndimage

class IntensityField:
    """An IntensityField represents the data collected from a DistributionField.  
    It consists of an array of intensities g. [nx x ny x nz x npol]
    """
    def __init__(self, g=None):
        self.g = g

    def load_from_file(self, file_names=None, x0=0, y0=0, z0=0,
                       xspan=10, yspan=10, zspan=10, cal=None,
                       angle=-35):

        # Rotate and make arrow
        # im = util.tiff2array(file_names[0], x=x0, y=y0, z=z0, width=xspan, height=yspan, slices=zspan)
        # im = scipy.ndimage.rotate(im, angle=angle, axes=(0, 2))
        g = np.zeros((xspan, yspan, zspan, len(file_names)))
        
        for i, file_name in enumerate(file_names):
            im = util.tiff2array(file_name, x=x0, y=y0, z=z0, width=xspan, height=yspan, slices=zspan)
            # im = scipy.ndimage.rotate(im, angle=angle, axes=(0, 2))
            if cal is None:
                g[:,:,:,i] = im
            else:
                g[:,:,:,i] = im/cal[i]
        self.g = g/g.max()
                
    def plot(self, output_file='out.pdf', shape=(2, 4),
             row_labels=None, col_labels=None,
             dpi=400, d=50, mag=1, show=False):
        # Create axes
        inches = 3
        fig, axs = plt.subplots(shape[0], shape[1]+1,
                                figsize=(shape[1]*inches, shape[0]*inches),
                                gridspec_kw={'hspace':-0.175, 'wspace':-0.075, 'width_ratios':[1, 1, 1, 1, 0.05]})
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))

        # Load data and plot
        for i, ax in enumerate(axs[:,:4].flatten()):
            
            # Calculate contours and plot
            mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
            mlab.clf()
            im = self.g[:,:,:,i]
            vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(im), vmin=0, vmax=.8, color=(1, 0, 0))

            # Changing the ctf:
            from tvtk.util.ctf import ColorTransferFunction

            ctf = ColorTransferFunction()
            for i in 0.1*np.arange(9):
                c = (1,1,1)#cm.binary(i)
                ctf.add_rgb_point(i, c[0], c[1], c[2]) 
            # ctf.add_hsv_point(value, h, s, v)
            # ...
            vol._volume_property.set_color(ctf)
            vol._ctf = ctf
            vol.update_ctf = True

            # Changing the otf:
            from tvtk.util.ctf import PiecewiseFunction
            otf = PiecewiseFunction()
            for i in 0.1*np.arange(9):
                otf.add_point(i, i**(3))
            vol._otf = otf
            vol._volume_property.set_scalar_opacity(otf)

            mlab.gcf().scene.parallel_projection = True
            mlab.points3d(0,0,0, color=(1, 1, 1))
            
            mlab.outline(extent=[0,im.shape[0],0,im.shape[1],0,im.shape[2]], line_width=2*mag)
            mlab.view(azimuth=225, elevation=45, distance=d, focalpoint=None,
                      roll=None, reset_roll=True, figure=None)

            if show:
                mlab.show()
            mlab.savefig('temp.png', magnification=mag)
            im = mpimg.imread('temp.png')
            ax.imshow(im, interpolation=None, vmin=0, vmax=1)            
            ax.set_axis_off()
        subprocess.call(['rm', 'temp.png'])

        # Plot colorbars
        import matplotlib as mpl
        axs[0,-1].set_axis_off()
        # norm = mpl.colors.Normalize(vmin=0, vmax=1)
        # cb1 = mpl.colorbar.ColorbarBase(axs[0,-1], cmap=cm.hot, norm=norm, orientation='vertical')
        # cb1.set_label('Color Transfer Function', rotation=270, labelpad=22, fontsize=14)
        # cb1.ax.tick_params(axis='both', labelsize=14)

        norm = mpl.colors.PowerNorm(gamma=1./3.)
        cb2 = mpl.colorbar.ColorbarBase(axs[1,-1], cmap=cm.binary, norm=norm, orientation='vertical')
        cb2.set_label('Intensity = Opacity', rotation=270, labelpad=22, fontsize=14)
        cb2.ax.tick_params(axis='both', labelsize=14)
        
        # Adjust colorbar position
        diff = 0.05
        pos1 = axs[0,-1].get_position()
        pos2 = [pos1.x0 + diff/5, pos1.y0 + diff,  pos1.width, pos1.height - 2*diff] 
        axs[0, -1].set_position(pos2)
        pos1 = axs[1,-1].get_position()
        pos2 = [pos1.x0 + diff/5, pos1.y0 + diff,  pos1.width, pos1.height - 2*diff] 
        axs[1, -1].set_position(pos2)

        # Add xyz axes
        from scipy.ndimage import imread
        
        axis1 = fig.add_axes([0.07, 0.775, 0.15, 0.15])
        axis1.imshow(imread('frames/frames_a.png'))
        axis1.set_axis_off()
        axis2 = fig.add_axes([0.07, 0.425, 0.15, 0.15])
        axis2.imshow(imread('frames/frames_b.png'))
        axis2.set_axis_off()
        
        # Label rows and columns
        if col_labels is not None:
            for i, label in enumerate(col_labels):
                axs[0, i].annotate(label, xy=(0,0), xytext=(0.5, 1.05), textcoords='axes fraction', va='center', ha='center', fontsize=14, annotation_clip=False)
        if row_labels is not None:
            for i, label in enumerate(row_labels):
                axs[i, 0].annotate(label, xy=(0,0), xytext=(-0.08, 0.5), textcoords='axes fraction', va='center', ha='center', fontsize=14, annotation_clip=False, rotation=90)

        fig.savefig(output_file, dpi=dpi)
