import numpy as np
from polharmonic import util, ill, det, sft

class Microscope:
    """
    A Microscope represents an experiment that collects a single frame of 
    intensity data.  

    A Microscope is specified by its illumination path (an Illuminator object),
    and its detection path (a Detector object).
    """
    def __init__(self, ill=ill.Illuminator(), det=det.Detector(),
                 color=(1,0,0)):
        self.ill = ill
        self.det = det
        self.prf = self.ill.exc_prf()*self.det.det_prf()
        self.color = color

    def Hlm(self, max_l=4):
        self.Hlm = sft.sft(self.prf, max_l=max_l)
        return self.Hlm

    def plot_micro(filename=None):
        # TODO 
        return 0

    def plot_scene(self, filename):
        util.draw_scene(self.scene_string(), filename=filename, save_file=True)

    def scene_string(self):
        asy_string = ''
        ill_string = "mydot(theta, color);\n"
        ill_string = ill_string.replace('theta', str(np.deg2rad(self.ill.theta_optical_axis)))
        ill_string = ill_string.replace('color', str(self.color))
        asy_string += ill_string

        pol_string = "arrow(theta, phi_pol, color, false);\n"
        pol_string = pol_string.replace('theta', str(self.ill.theta_optical_axis))
        pol_string = pol_string.replace('phi_pol', str(self.ill.phi_pol))
        pol_string = pol_string.replace('color', str(self.color))
        asy_string += pol_string
        
        det_string = "circle(theta, alpha, true, color);\n"
        det_string = det_string.replace('theta', str(self.det.theta_optical_axis))
        det_string = det_string.replace('alpha', str(np.arcsin(self.det.NA/self.det.n)))
        det_string = det_string.replace('color', str(self.color))
        asy_string += det_string
        return asy_string
