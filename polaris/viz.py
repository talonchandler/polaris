from scipy import misc
from polaris import util
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import vtk
from dipy.viz import window, actor

def plot5d(filename, data, row_labels=None, col_labels=None, yscale_label=None,
           force_bwr=False, normalize=False):
    if np.min(data) < 0 or force_bwr:
        colormap = 'bwr'
        vmin = -1
        vmax = 1
    else:
        colormap = 'gray'
        vmin = 0
        vmax = 1

    if normalize:
        data = data/np.max(data)
    
    inches = 2
    rows = data.shape[-1]
    cols = data.shape[-2] + 1
    widths = [1]*(cols - 1) + [0.05]
    heights = [1]*rows
    f = plt.figure(figsize=(inches*np.sum(widths), inches*np.sum(heights)))
    spec = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths,
                             height_ratios=heights, hspace=0.25, wspace=0.15)
    for row in range(rows):
        for col in range(cols):
            if col != cols - 1:
                data3 = data[:,:,:,col,row]
                plot_projections(data3, f, spec, row, col, col_labels, row_labels,
                                 vmin, vmax, colormap, cols, yscale_label)
            elif col == cols - 1 and row == rows - 1:
                plot_colorbar(f, spec, row, col, vmin, vmax, colormap)

    f.savefig(filename, bbox_inches='tight')

def plot_projections(data3, f, spec, row, col, col_labels, row_labels, vmin, vmax,
                     colormap, cols, yscale_label, pos=(-0.05, 1.05, 0.5, 0.5)):
    mini_spec = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=spec[row, col], hspace=0.1, wspace=0.1)
    for a in range(2):
        for b in range(2):
            ax = f.add_subplot(mini_spec[a, b])
            if a == 0 and b == 0:
                data2 = util.absmax(data3, axis=0)[:,::-1]
            if a == 0 and b == 1:
                data2 = util.absmax(data3, axis=2).T
            if a == 1 and b == 1:
                data2 = util.absmax(data3, axis=1)[:,::-1].T
                draw_annotations(ax, row, col, row_labels, col_labels, pos=pos)
            if a == 1 and b == 0:
                if vmin == -1:
                    data2 = np.zeros(data3.shape[0:2])
                else:
                    data2 = np.ones(data3.shape[0:2])
            ax.imshow(data2, cmap=colormap, vmin=vmin, vmax=vmax, interpolation='none', origin='lower', extent=[-24, 24, -24, 24], aspect=1)
            ax.axis('off')
            if col == (cols - 2) and row == 0 and a == 0 and b == 1 and yscale_label is not None:
                ax.annotate(yscale_label, xy=(1.3,0.5), xytext=(1.3, 0.5), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=10, rotation=-90)
                ax.annotate('', xy=(1.1,0), xytext=(1.1, 1), xycoords='axes fraction', textcoords='axes fraction', va='center', arrowprops=dict(arrowstyle='|-|, widthA=0.2, widthB=0.2', shrinkA=0.05, shrinkB=0.05, lw=0.5))

def plot_images(images, f, spec, row, col, col_labels, row_labels, vmin, vmax,
                colormap, cols, yscale_label, pos=(-0.05, 1.05, 0.5, 0.5)):
    mini_spec = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=spec[row, col], hspace=0.1, wspace=0.1)
    for a in range(2):
        for b in range(2):
            ax = f.add_subplot(mini_spec[a, b])
            if a == 0 and b == 0:
                image = images[0]
            if a == 0 and b == 1:
                image = images[1]
            if a == 1 and b == 1:
                image = images[2]
                draw_annotations(ax, row, col, row_labels, col_labels, pos=pos)
            if a == 1 and b == 0:
                image = None
            if image is not None:
                im = misc.imread(image)
                ax.imshow(im, interpolation='none', origin='upper', extent=[-24, 24, -24, 24], aspect=1)
            ax.axis('off')
            if col == (cols - 2) and row == 0 and a == 0 and b == 1 and yscale_label is not None:
                ax.annotate(yscale_label, xy=(1.3,0.5), xytext=(1.3, 0.5), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=10, rotation=-90)
                ax.annotate('', xy=(1.1,0), xytext=(1.1, 1), xycoords='axes fraction', textcoords='axes fraction', va='center', arrowprops=dict(arrowstyle='|-|, widthA=0.2, widthB=0.2', shrinkA=0.05, shrinkB=0.05, lw=0.5))
                
def plot_colorbar(f, spec, row, col, vmin, vmax, colormap):
    ax = f.add_subplot(spec[row, col])
    X, Y = np.meshgrid(np.linspace(vmin, vmax, 100),
                       np.linspace(vmin, vmax, 100))
    ax.imshow(Y, cmap=colormap, vmin=vmin, vmax=vmax, interpolation='none',
              extent=[vmin,vmax,vmin,vmax], origin='lower', aspect='auto')
    ax.set_xlim([vmin,vmax])
    ax.set_ylim([vmin,vmax])
    ax.tick_params(direction='out', left=False, right=True)
    ax.xaxis.set_ticks([])
    ax.yaxis.tick_right()
    if vmin == -1:
        ax.yaxis.set_ticks([-1.0, 0, 1.0])
    else:
        ax.yaxis.set_ticks([0, 0.5, 1.0])

def draw_annotations(ax, row, col, row_labels, col_labels, pos=(-0.05, 1.05, 0.5, 0.5)):
    xc = pos[0]
    yc = pos[1]
    d1 = pos[2]
    d2 = pos[3]
    ax.annotate('', xy=(xc,yc), xytext=(xc+d1, yc), xycoords='axes fraction', textcoords='axes fraction', va='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
    ax.annotate('', xy=(xc,yc), xytext=(xc-d1, yc), xycoords='axes fraction', textcoords='axes fraction', ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
    ax.annotate('', xy=(xc,yc), xytext=(xc, yc+d1), xycoords='axes fraction', textcoords='axes fraction', ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
    ax.annotate('', xy=(xc,yc), xytext=(xc, yc-d1), xycoords='axes fraction', textcoords='axes fraction', va='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
    ax.annotate('$x$', xy=(xc,yc), xytext=(xc+d2, yc), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=6)
    ax.annotate('$z$', xy=(xc,yc), xytext=(xc-d2, yc), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=6)
    ax.annotate('$y$', xy=(xc,yc), xytext=(xc, yc+d2), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=6)
    ax.annotate('$z$', xy=(xc,yc), xytext=(xc, yc-d2), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=6)
    if col_labels is not None:
        ax.annotate(col_labels[row,col], xy=(xc,yc), xytext=(0, 2.3), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=10)
    if col == 0 and row_labels is not None:
        ax.annotate(row_labels[row], xy=(xc,yc), xytext=(-1.3, 1), xycoords='axes fraction', textcoords='axes fraction', va='center', ha='center', fontsize=10, rotation=90)


def odf_slicer(odfs, affine=None, mask=None, sphere=None, scale=2.2,
               norm=True, radial_scale=True, opacity=1.,
               colormap='plasma', global_cm=False):
    if mask is None:
        mask = np.ones(odfs.shape[:3], dtype=np.bool)
    else:
        mask = mask.astype(np.bool)

    szx, szy, szz = odfs.shape[:3]

    class OdfSlicerActor(vtk.vtkLODActor):

        def display_extent(self, x1, x2, y1, y2, z1, z2):
            tmp_mask = np.zeros(odfs.shape[:3], dtype=np.bool)
            tmp_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = True
            tmp_mask = np.bitwise_and(tmp_mask, mask)

            self.mapper = actor._odf_slicer_mapper(odfs=odfs,
                                             affine=affine,
                                             mask=tmp_mask,
                                             sphere=sphere,
                                             scale=scale,
                                             norm=norm,
                                             radial_scale=radial_scale,
                                             opacity=opacity,
                                             colormap=colormap,
                                             global_cm=global_cm)
            self.SetMapper(self.mapper)

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(0, szx - 1, 0, szy - 1, 0, szz - 1)
                # self.display_extent(0, szx - 1, 0, szy - 1,
                #                     int(np.floor(szz/2)), int(np.floor(szz/2)))

            if x is not None:
                self.display_extent(x, x, 0, szy - 1, 0, szz - 1)
            if y is not None:
                self.display_extent(0, szx - 1, y, y, 0, szz - 1)
            if z is not None:
                self.display_extent(0, szx - 1, 0, szy - 1, z, z)

    odf_actor = OdfSlicerActor()
    odf_actor.display_extent(0, szx - 1, 0, szy - 1, 0, szz - 1)

    return odf_actor

def peak_slicer(peaks_dirs, peaks_values=None, mask=None, affine=None,
                colors=(1, 0, 0), opacity=1., linewidth=1,
                lod=False, lod_points=10 ** 4, lod_points_size=3):
    peaks_dirs = np.asarray(peaks_dirs)
    if peaks_dirs.ndim > 5:
        raise ValueError("Wrong shape")

    peaks_dirs = actor._makeNd(peaks_dirs, 5)
    if peaks_values is not None:
        peaks_values = actor._makeNd(peaks_values, 4)

    grid_shape = np.array(peaks_dirs.shape[:3])

    if mask is None:
        mask = np.ones(grid_shape).astype(np.bool)

    class PeakSlicerActor(vtk.vtkLODActor):

        def display_extent(self, x1, x2, y1, y2, z1, z2):

            tmp_mask = np.zeros(grid_shape, dtype=np.bool)
            tmp_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = True
            tmp_mask = np.bitwise_and(tmp_mask, mask)

            ijk = np.ascontiguousarray(np.array(np.nonzero(tmp_mask)).T)
            if len(ijk) == 0:
                self.SetMapper(None)
                return
            if affine is not None:
                ijk_trans = np.ascontiguousarray(apply_affine(affine, ijk))
            list_dirs = []
            for index, center in enumerate(ijk):
                # center = tuple(center)
                if affine is None:
                    xyz = center[:, None]
                else:
                    xyz = ijk_trans[index][:, None]
                xyz = xyz.T
                for i in range(peaks_dirs[tuple(center)].shape[-2]):

                    if peaks_values is not None:
                        pv = peaks_values[tuple(center)][i]
                    else:
                        pv = 1.
                    symm = np.vstack((-peaks_dirs[tuple(center)][i] * pv + xyz,
                                      peaks_dirs[tuple(center)][i] * pv + xyz))
                    list_dirs.append(symm)

            self.mapper = actor.line(list_dirs, colors=colors,
                                     opacity=opacity, linewidth=linewidth,
                                     lod=lod, lod_points=lod_points,
                                     lod_points_size=lod_points_size).GetMapper()
            self.SetMapper(self.mapper)

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(0, szx - 1, 0, szy - 1, 0, szz - 1)
            if x is not None:
                self.display_extent(x, x, 0, szy - 1, 0, szz - 1)
            if y is not None:
                self.display_extent(0, szx - 1, y, y, 0, szz - 1)
            if z is not None:
                self.display_extent(0, szx - 1, 0, szy - 1, z, z)

    peak_actor = PeakSlicerActor()

    szx, szy, szz = grid_shape
    peak_actor.display_extent(0, szx - 1, 0, szy - 1, 0, szz - 1)

    return peak_actor
