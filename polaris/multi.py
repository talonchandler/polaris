from polharmonic import ill, det, micro, util, dist, sft, data
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import sys
import numpy as np

class MultiMicroscope:
    """A MultiMicroscope represents an experiment that collects intensity data 
    under several different conditions (different polarization states or 
    illumination schemes).

    A MultiMicroscope mainly consists of a list of Microscopes.
    """
    def __init__(self, ill_thetas=[0], det_thetas=[0],
                 det_nas=[0.8], n_samp=1.33,
                 phi_pols=[0, 45, 90, 135],
                 max_l=4, n_pts=100):
        
        m = [] # List of microscopes

        # Cycle through illumination/detection paths
        for i, det_theta in enumerate(det_thetas):
            # Cycle through illumination polarizations
            for phi_pol in phi_pols:
                ill_ = ill.Illuminator(theta_optical_axis=ill_thetas[i],
                                       phi_pol=phi_pol)
                det_ = det.Detector(theta_optical_axis=det_thetas[i],
                                    NA=det_nas[i], n=1.33)
                m.append(micro.Microscope(ill=ill_, det=det_)) # Add microscope

        self.micros = m
        self.max_l = max_l
        self.max_j = int((max_l + 1)*(max_l + 2)/2)
        self.n_pts = n_pts

    def calc_sys_matrix(self):
        psi = []
        for micro in self.micros:
            psi_row = sft.sft(micro.prf, max_l=self.max_l)
            psi.append(psi_row)
        psi = np.array(psi, dtype=float)
        psi[np.abs(psi) < 1e-15] = 0
        self.psi = psi

    def calc_B_matrix(self):
        # B is a discrete inverse spherical Fourier transform
        # f = BF, F is max_j x 1, B is n_pts x max_j, f is n_pts x 1

        # Find fibonacci points
        pts = util.fibonacci_sphere(self.n_pts)
        B = np.zeros((self.n_pts, self.max_j))

        # Calculate B
        for index, x in np.ndenumerate(B):
            theta, phi = pts[index[0]]            
            l, m = util.j2lm(index[1])
            B[index] = util.spZnm(l, m, theta, phi)
        
        self.B = B

        # Calculate x,y,z spherical coordinates and triangulation
        theta = pts[:,0]
        phi = pts[:,1]
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)
        self.xyz = np.vstack([x,y,z]).T
        
        from scipy.spatial import ConvexHull
        ch = ConvexHull(self.xyz)
        self.triangles = ch.simplices

    def calc_intensity_dist(self, dist):
        return np.matmul(self.psi, dist.sh)

    def calc_intensity_field(self, dist_field):
        g = np.einsum('ij,klmj->klmi', self.psi, dist_field.sh_arr)
        intf = data.IntensityField()
        intf.g = g
        return intf
    
    def recon_dist(self, g, prior=None):
        if prior is None:
            N = self.B.shape[1]
            M = self.B.shape[0]
            P = matrix(2*np.matmul(self.psi.T, self.psi), tc='d')
            q = matrix(-2*np.matmul(g, self.psi), tc='d')
            G = matrix(-self.B, tc='d')
            h = matrix(np.zeros(M), tc='d')
            sol = solvers.qp(P, q, G, h)
            sh = np.array(sol['x']).flatten()
            d = dist.Distribution(sh=sh)
            return d
        elif prior is 'single':
            # Construct prior set (single directions w/ 10 amplitudes)
            f_prior = np.hstack([(x*0.1 + 0.1)*np.identity(self.B.shape[0]) for x in range(10)])
            H_model = np.matmul(self.psi, np.linalg.pinv(self.B))
            g_model = np.matmul(H_model, f_prior)
            g_model = g_model/np.max(g_model)
            g_diff = g_model - g[:, np.newaxis]
            obj = np.linalg.norm(g_diff, ord=2, axis=0)**2
            argmin = np.argmin(obj)
            f = f_prior[:, argmin]
            d = dist.Distribution(f=f)
            return d

    def recon_dist_field(self, intf, mask=None, prior=None):
        g = intf.g
        N = np.sum(mask)
        j = 1        
        if prior is None:
            dist_arr = np.zeros([*g.shape[:-1], self.max_j])
            for i in np.ndindex(g.shape[:-1]):
                if mask[i]:
                    sys.stdout.flush()
                    sys.stdout.write("Reconstructing: "+ str(j) + '/' + str(N) + '\r')
                    j += 1
                    d = self.recon_dist(g[i], prior=prior)
                    dist_arr[i] = d.sh
                else:
                    dist_arr[i] = np.zeros(self.max_j)
            d = dist.DistributionField(sh_arr=dist_arr)
            d.calc_f_arr(self.B)
            return d
        
        elif prior is 'single':
            dist_arr = np.zeros([*g.shape[:-1], self.B.shape[0]])
            mask_idx = np.nonzero(mask)
            for i in range(mask_idx[0].shape[0]):
                sys.stdout.flush()
                sys.stdout.write("Reconstructing: "+ str(j) + '/' + str(N) + '\r')
                j += 1
                idx = mask_idx[0][i], mask_idx[1][i], mask_idx[2][i]
                d = self.recon_dist(g[idx], prior=prior)
                dist_arr[idx] = d.f
            return dist.DistributionField(f_arr=dist_arr)

    def plot_scene(self, filename):
        scene_string = ''
        for micro in self.micros:
            scene_string += micro.scene_string()
        util.draw_scene(scene_string, filename=filename, save_file=True)

    def plot_scenes(self, file_prefix, skip_sch=True):
        sch_tex_string = ''
        for i, micro in enumerate(self.micros):
            filename = file_prefix + str(i) + '.png'
            print('Plotting ' + filename + '...')
            if not skip_sch:
                util.draw_scene(micro.scene_string(), filename=filename, save_file=True)
            tex_string = '\picbig{2.0}{'+file_prefix.split('/')[-1]+'XXX}\\\\'
            sch_tex_string += tex_string.replace('XXX', str(i))
        return sch_tex_string

    def plot_sh(self, file_prefix, string_mask=None):
        sh_tex_string = ''
        for i in range(self.max_j):
            filename = file_prefix+str(i)+'.png'
            print('Plotting ' + filename + '...')
            sh = np.zeros(self.max_j)
            sh[i] = 1
            d = dist.Distribution(sh)
            d.plot_dist(self.B, self.xyz, self.triangles, filename=filename)
            tex_string = '\pic{1.0}{'+file_prefix.split('/')[-1]+'XXX}\\\\'
            if string_mask is not None:
                if not string_mask[i]:
                    sh_tex_string += tex_string.replace('XXX', str(i))
            else:
                sh_tex_string += tex_string.replace('XXX', str(i))
        return sh_tex_string

    def plot_matrix(self, folder, skip_sch=False):
        zero_mask = np.all(np.abs(self.psi) < 1e-10, axis=0)
        masked_psi = self.psi[:, ~zero_mask]
        
        sh_tex_string = self.plot_sh(folder+'/sh', string_mask=zero_mask)
        sch_tex_string = self.plot_scenes(folder+'/sch', skip_sch)
        util.create_latex_matrix(sch_tex_string, sh_tex_string, masked_psi, folder+'/mat')

    def plot_svs(self, folder):
        # SVD
        U, s, Vh = np.linalg.svd(self.psi)

        # Plot individual singular distributions
        for i in range(np.sum(s > 1e-10)):
            d = dist.Distribution(Vh[i,:])
            filename = folder+'/sv'+str(i)+'.png'
            d.plot_dist(self.B, self.xyz, self.triangles, filename=filename)

        # TODO Automate svs
