import numpy as np
from polharmonic import util, sft
from mayavi import mlab
import subprocess
from cvxopt import matrix, solvers
import sys
import vispy
import scipy.misc

class DistributionField:
    """A DistributionField represents many fluorophore distributions. It 
    consists of an array of Distributions."""
    def __init__(self, sh_arr=None, f_arr=None):
        if f_arr is not None and sh_arr is not None:
            print("Warning: sh_arr and f_arr are redundant.")
        elif f_arr is None:
            self.sh_arr = sh_arr
            self.f_arr = None
        elif sh_arr is None:
            self.sh_arr = None
            self.f_arr = f_arr

    def calc_f_arr(self, B):
        self.f_arr = np.einsum('ij,klmj->klmi', B, self.sh_arr)

    def make_positive(self, B, max_l=None):
        for i in np.ndindex(self.sh_arr.shape[:2]):
            d = Distribution(self.sh_arr[i])
            d.make_positive(B, max_l=max_l)
            self.sh_arr[i] = d.sh
            
    def plot_dist_field(self, B, xyz, triangles, filename=None,
                        d=50, r=1, s=1, mag=1, show=False, mask=None,
                        vis_px=500, dpi=500):

        # Calculate radii
        radii = r*self.f_arr
        
        N = np.sum(mask)
        j = 1

        nz_idx = np.nonzero(np.sum(radii, axis=3))
        nz = radii[nz_idx]

        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
        mlab.clf()
                    
        if np.nonzero(nz)[0].shape[0] == nz.shape[0]:
            x, y, z, tp = np.nonzero(radii)
            u = xyz[tp, 0]
            v = xyz[tp, 1]
            w = xyz[tp, 2]

            # u = np.random.random(u.shape)
            # v = np.random.random(v.shape)
            # w = np.random.random(w.shape)
            
            scale = util.uvw2color(u, v, w) # color by direction

            #s = radii[x, y, z, tp] # color by magnitude
            
            quiv = mlab.quiver3d(x, y, z, u, v, w, scalars=scale, mode='cylinder',
                                 scale_factor=r, scale_mode='none', vmin=0, vmax=255)

            quiv.glyph.color_mode = 'color_by_scalar'
            
            quiv.glyph.glyph_source.glyph_source.center = [0, 0, 0]

            lut = quiv.module_manager.scalar_lut_manager.lut.table.to_array()
            lut = util.sphere_LUT()
            # lut[:, 0] = np.linspace(0, 255, 256)
            # lut[:, 1] = 0
            # lut[:, 2] = 0
            # lut[:, 3] = 255
            quiv.module_manager.scalar_lut_manager.lut.table = lut
            
            # xs, ys, zs, fs= self.f_arr.shape            
            # x = x - xs/2
            # y = y - ys/2
            # z = z - zs/2
            
            
            # vispy.use('PyQt4')
            # canvas = vispy.scene.SceneCanvas(keys='interactive', bgcolor='white',
            #                          size=(vis_px, vis_px), show=show, dpi=dpi)
            # my_cam = vispy.scene.cameras.turntable.TurntableCamera(fov=0, elevation=40, azimuth=135,
            #                                                        scale_factor=100)

            # view = canvas.central_widget.add_view(camera=my_cam)


            # dots = vispy.scene.visuals.Markers(parent=view.scene)
            # dots.antialias = False
            # dots.set_data(pos=np.array([[0, 0, 0], [1.01,0,0],[0,1.01,0],[0,0,1.01]]),
            #       edge_color='black', face_color='black', size=vis_px/50)

            # line = vispy.scene.visuals.Line(parent=view.scene)
            # #dots.antialias = False
            # line.set_data(pos=np.array([[1.01,0,0],[0,1.01,0],[0,0,1.01]]),
            #               color=np.array([[1, 0, 0, 0.5], [0, 1, 0, 1], [0, 0, 1,0]]))

            # Plot single line for speed
            # pos = []
            # color = []
            # for i, xi in enumerate(x):
            #     sys.stdout.flush()
            #     sys.stdout.write("Plotting: "+ str(i) + '/' + str(N) + '\r')
            #     e = 1e-3
            #     pos.append([x[i],y[i],z[i]])
            #     pos.append([x[i]+e,y[i]+e,z[i]+e])
            #     pos.append([x[i]+r*u[i],y[i]+r*v[i],z[i]+r*w[i]])
            #     pos.append([x[i]+r*u[i]+e,y[i]+r*v[i]+e,z[i]+r*w[i]+e])
            #     color.append([np.abs(u[i]), np.abs(v[i]), np.abs(w[i]), 0])
            #     color.append([np.abs(u[i]), np.abs(v[i]), np.abs(w[i]), 1])
            #     color.append([np.abs(u[i]), np.abs(v[i]), np.abs(w[i]), 1])
            #     color.append([np.abs(u[i]), np.abs(v[i]), np.abs(w[i]), 0])

            # line = vispy.scene.visuals.Line(parent=view.scene)
            # line.set_data(pos=np.array(pos), color=np.array(color), width=1)
            # tube = vispy.scene.visuals.Tube(parent=view.scene, points=np.array(pos), color=np.array(color), radius=.1)
           
            
            # im = canvas.render()
            # scipy.misc.imsave(filename, im)
            # if show:
            #     vispy.app.run()


            # xyz, tp = util.fibonacci_sphere(1000, xyz=True)
            # util.plot_sphere(filename='scale.png', directions=tp, data=xyz,
            #                  show=show, vis_px=500)

            # quiv = mlab.quiver3d(x, y, z, u, v, w, scalars=scale, scale_factor=r, mode='cylinder')
                                   
            # quiv.glyph.color_mode = 'color_by_scalar'
            # quiv.glyph.glyph_source.glyph_source.center = [0, 0, 0]
            # lut = quiv.module_manager.scalar_lut_manager.lut.table.to_array()
            # lut[:, -1] = np.linspace(0, 255, 256)
            # lut = util.sphere_LUT()
            # quiv.module_manager.scalar_lut_manager.lut.table = lut

            # Plot glyphs
            # for i, xi in enumerate(x):
            #     sys.stdout.flush()            
            #     sys.stdout.write("Plotting: "+ str(i) + '/' + str(N) + '\r')
            #     mlab.plot3d([x[i], x[i] + r*u[i]], [y[i], y[i] + r*v[i]], [z[i], z[i] + r*w[i]],
            #                 color=(np.abs(u[i]), np.abs(v[i]), np.abs(w[i])),
            #                 representation='surface', tube_radius=0.3,
            #                 tube_sides=12)
        if radii.shape[2] != 1: # If 3D dataset
            extent = [0, radii.shape[0], 0, radii.shape[1], 0, radii.shape[2]]
            mlab.points3d(0,0,0, color=(1, 1, 1))
            mlab.outline(extent=extent, line_width=2*mag)            

        # View and save
        mlab.gcf().scene.parallel_projection = True
        mlab.view(azimuth=225, elevation=45, distance=d, focalpoint=None,
                  roll=None, reset_roll=True, figure=None)
        mlab.savefig(filename, magnification=mag)        
        if show:
            mlab.show()
                
            
        # else:
        #     # TODO replace for loop with single tri_mesh
        #     for i in range(nz.shape[0]):
        #         sys.stdout.flush()            
        #         sys.stdout.write("Plotting: "+ str(j) + '/' + str(N) + '\r')
        #         idx = [x[i] for x in nz_idx]
        #         r = nz[i,:]
        #         j += 1
        #         mlab.triangular_mesh(r*xyz[:,0] + s*idx[0],
        #                              r*xyz[:,1] + s*idx[1],
        #                              r*xyz[:,2] + s*idx[2],
        #                              triangles, color=(1, 0, 0))

    def plot_dist_field_color(self, B, xyz, triangles, filename=None,
                              d=50, r=1, s=1, mag=1, show=False, mask=None,
                              vis_px=500, dpi=500, gmask=None, rmask=None):

        # Calculate radii
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
        mlab.clf()

        radii = r*self.f_arr
        radii_r = np.copy(radii)
        radii_g = np.copy(radii)

        x, y, z, tp = np.nonzero(radii)
        u = xyz[tp, 0]
        v = xyz[tp, 1]
        w = xyz[tp, 2]

        for i, rad in enumerate(x):
            print(i)
            if -0.7*x[i] + z[i] > -7:
                quiv = mlab.quiver3d(x[i], y[i], z[i], u[i], v[i], w[i], color=(1, 0, 0), mode='cylinder',
                                     scale_factor=r, scale_mode='none', vmin=0, vmax=255)
            else:
                quiv = mlab.quiver3d(x[i], y[i], z[i], u[i], v[i], w[i], color=(0, 1, 0), mode='cylinder',
                                     scale_factor=r, scale_mode='none', vmin=0, vmax=255)


        if radii.shape[2] != 1: # If 3D dataset
            extent = [0, radii.shape[0], 0, radii.shape[1], 0, radii.shape[2]]
            mlab.points3d(0,0,0, color=(1, 1, 1))
            mlab.outline(extent=extent, line_width=2*mag)            

        # View and save
        mlab.gcf().scene.parallel_projection = True
        mlab.view(azimuth=225, elevation=45, distance=d, focalpoint=None,
                  roll=None, reset_roll=True, figure=None)
        mlab.savefig(filename, magnification=mag)        
        if show:
            mlab.show()
        


class Distribution:
    """A Distribution represents a fluorophore distribution. It has redundant
    representations in the angular domain (orientation distribution function)
    and the angular frequency domain (spherical harmonic coefficients).

    """
    def __init__(self, sh=None, f=None):
        if f is None:
            self.sh = sh
        if sh is None:
            self.f = f

    def make_positive(self, B, max_l=None):
        N = B.shape[1]
        M = B.shape[0]
        P = matrix(2*np.identity(N), tc='d')
        q = matrix(-2*self.sh, tc='d')
        G = matrix(-B, tc='d')
        h = matrix(np.zeros(M), tc='d')
        if max_l is None:
            sol = solvers.qp(P, q, G, h)
        else:
            J = util.maxl2maxj(max_l)
            A = matrix(np.identity(N)[N-J:,:], tc='d')
            b = matrix(np.zeros(J), tc='d')
            sol = solvers.qp(P, q, G, h, A, b)
        self.sh = np.array(sol['x']).flatten()
        
    def plot_dist(self, B, xyz, triangles, filename=None, r=1, mag=1, show=False):
        # Calculate radii
        radii = np.matmul(B, self.sh)

        # Split into positive and negatives
        n = radii.clip(max=0) 
        p = radii.clip(min=0)*(-1)

        # Create figure
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
        mlab.clf()
        
        # Plot
        mlab.triangular_mesh(p*xyz[:,0], p*xyz[:,1], p*xyz[:,2], triangles, color=(1, 0, 0))
        mlab.triangular_mesh(n*xyz[:,0], n*xyz[:,1], n*xyz[:,2], triangles, color=(0, 0, 1))
        
        # View and save
        mlab.view(azimuth=135, elevation=45, distance=3, focalpoint=None,
                  roll=None, reset_roll=True, figure=None)
        mlab.savefig(filename, magnification=mag)
        subprocess.call(['convert', filename, '-transparent', 'white', filename])
        if show:
            mlab.show()
