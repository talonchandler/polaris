# Custom vispy classes
from polharmonic import util
import numpy as np
from vispy.geometry import create_sphere, MeshData
from vispy.visuals.mesh import MeshVisual
from vispy.visuals import CompoundVisual
from vispy.scene.visuals import create_visual_node
import matplotlib
import matplotlib.pyplot as plt
from vispy.visuals.transforms import (STTransform, LogTransform,
                                      MatrixTransform, PolarTransform)
from vispy.visuals.line import LineVisual
from vispy.visuals.text import TextVisual

class MySphereVisual(CompoundVisual):
    
    def __init__(self, radius=1.0, directions=None, colors=None):
        # Convert spherical to cartesian
        points = np.array([util.tp2xyz(*x) for x in directions])

        # Create mesh
        import scipy.spatial
        ch = scipy.spatial.ConvexHull(points)
        mesh = MeshData(vertices=ch.points, faces=ch.simplices)

        self._mesh = MeshVisual(vertices=mesh.get_vertices(),
                                faces=mesh.get_faces(),
                                vertex_colors=colors)

        CompoundVisual.__init__(self, [self._mesh])
        self.mesh.set_gl_state(depth_test=True)

    @property
    def mesh(self):
        """The vispy.visuals.MeshVisual that used to fil in.
        """
        return self._mesh

MySphere = create_visual_node(MySphereVisual)

class MyXYZAxisVisual(CompoundVisual):
    """
    Simple 3D axis for indicating coordinate system orientation. Axes are
    x=red, y=green, z=blue.
    """
    def __init__(self, origin=[0,0,0], length=1):
        verts = origin + np.array([[0, 0, 0],
                                   [length, 0, 0],
                                   [0, 0, 0],
                                   [0, length, 0],
                                   [0, 0, 0],
                                   [0, 0, length]])

        line = LineVisual(pos=verts, color=np.array([0, 0, 0, 1]),
                          connect='segments', method='gl')

        x = TextVisual('x', font_size=12, pos=origin + np.array([1.25*length,0,0]))
        y = TextVisual('y', font_size=12, pos=origin + np.array([0,1.25*length,0]))
        z = TextVisual('z', font_size=12, pos=origin + np.array([0,0,1.25*length]))

        CompoundVisual.__init__(self, [line, x, y, z])

MyXYZAxis = create_visual_node(MyXYZAxisVisual)        
