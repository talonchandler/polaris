#!/usr/bin/env pvpython

# Imports
import argparse
import logging
import datetime
import subprocess
import sys
import os
import time; start = time.time();

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-filename', 
                    default=None, help='Specify the data filename.')

parser.add_argument('-l', '--log-filename', default='polvis.log', 
                    help='Specify the log filename.')

parser.add_argument('-o', '--output-filename', default='output.png',
                    help='Specify the output filename. Choose from output types [.avi, .mp4, .png, .]. Dot suffix outputs to folder of .pngs.')

parser.add_argument('-m', '--magnification', default=1, 
                    help='Output magnification. Modifies dpi screenshots. Use --size for animations.')

parser.add_argument('-n', '--n_frames', default=30, type=int,
                    help='Number of frames. Only works if output file is [.avi, .mp4]')

parser.add_argument('-v', '--view', action='store_true',
                    help='View output file in VLC.app.')

parser.add_argument('-pv', '--paraview', action='store_true',
                    help='View output in Paraview.')

parser.add_argument('--fps', default=3, type=int,
                    help='Frame rate. Only works if output file is [.avi, .mp4]')

parser.add_argument('--size', nargs=2, type=int, default=[1000, 700],
                    help='Image size in pixels. Only works for animations [.avi, .mp4]. Use -m for screenshots.')

arg = parser.parse_args()

if arg.input_filename is None:
    arg.input_filename = sys.stdin.readline()

# Setup logging
log = logging.getLogger('vis')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)
fh = logging.FileHandler(arg.log_filename, mode='w')
fh.setLevel(logging.DEBUG) 
log.addHandler(fh)

# Log basics
log.info('-----polvis-----')
log.info('Platform:\t'+sys.platform)
v = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                     cwd='/'.join(__file__.split('/')[:-1])+'/',
                     stdout=subprocess.PIPE).communicate()[0]
log.info('Version:\t'+v.strip().decode('ascii'))
log.info('Time:\t\t'+datetime.datetime.now().replace(microsecond=0).isoformat())
log.info('Command:\t'+' '.join(sys.argv))

# Main script
log.info('-----setting up ParaView scene-----')
from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

# Create a new 'Render View'
view = CreateView('RenderView')
view.ViewSize = arg.size
view.AnnotationColor = [0.0, 0.0, 0.0]
view.AxesGrid = 'GridAxes3DActor'
view.OrientationAxesVisibility = 0
view.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
view.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
view.CenterOfRotation = [43.5, 49.5, 19.5]
view.StereoType = 0
view.CameraPosition = [-92.82, -99.84, 104.75]
view.CameraFocalPoint = [43.5, 49.5, 19.5]
view.CameraViewUp = [0.36, 0.18, 0.91]
view.CameraParallelScale = 68.72
view.Background = [1.0, 1.0, 1.0]

# Create a new 'XML Unstructured Grid Reader'
data = XMLUnstructuredGridReader(FileName=[arg.input_filename])
data.PointArrayStatus = ['r', 'uvw']

# Show data from data
dataDisplay = Show(data, view)

# Glyph settings
dataDisplay.Representation = '3D Glyphs'
dataDisplay.DiffuseColor = [0.66, 0.0, 0.0]
dataDisplay.GlyphType = 'Arrow'
dataDisplay.ScaleFactor = 3.0
dataDisplay.GlyphType.TipResolution = 1
dataDisplay.GlyphType.TipRadius = 0.0
dataDisplay.GlyphType.TipLength = 0.0
dataDisplay.GlyphType.ShaftResolution = 32
dataDisplay.GlyphType.ShaftRadius = 0.1
dataDisplay.Orient = 1
dataDisplay.SelectOrientationVectors = 'uvw'
dataDisplay.Scaling = 1
dataDisplay.ScaleMode = 'Magnitude'
dataDisplay.SelectScaleArray = 'r'

# Outline
data_1 = XMLUnstructuredGridReader(FileName=[arg.input_filename])
data_1.PointArrayStatus = ['r', 'uvw']
data_1Display = Show(data_1, view)
data_1Display.Representation = 'Outline'
data_1Display.AmbientColor = [0.0, 0.0, 0.0]

# Create animation
animationScene1 = GetAnimationScene()
animationScene1.NumberOfFrames = arg.n_frames
anim = GetCameraTrack(view=view)

# Create keyframes for orbit
# Create a key frame
kf1 = CameraKeyFrame()
kf1.Position = [-95.71, -98.17, 69.22]
kf1.FocalPoint = [43.5, 49.5, 19.5]
kf1.ViewUp = [0.18, 0.14, 0.97]
kf1.ParallelScale = 68.72
kf1.PositionPathPoints = [-95.71, -98.17, 69.22, 73.01, -155.76, 45.10, 219.86,
                          -61.18, 1.99, 235.96, 115.45, -28.13, 109.38, 243.19,
                          -22.94, -66.04, 227.34, 13.70, -160.25, 79.64, 54.65]
kf1.FocalPathPoints = [43.5, 49.5, 19.5]
kf1.ClosedPositionPath = 1

# Create a key frame
kf2 = CameraKeyFrame()
kf2.KeyTime = 1.0
kf2.Position = [-95.71, -98.17, 69.22]
kf2.FocalPoint = [43.5, 49.5, 19.5]
kf2.ViewUp = [0.18, 0.14, 0.97]
kf2.ParallelScale = 68.72

# Initialize the animation track
anim.Mode = 'Path-based'
anim.KeyFrames = [kf1, kf2]

# Current camera placement for renderView1
view.CameraPosition = [-95.71, -98.17, 69.22]
view.CameraFocalPoint = [43.5, 49.5, 19.5]
view.CameraViewUp = [0.18, 0.14, 0.97]
view.CameraParallelScale = 68.72

log.info('-----saving output-----')
file_name = arg.output_filename.split('.')[0]
file_type = arg.output_filename.split('.')[-1]

if file_type in ['avi', 'mp4', '']:
    log.info('Rendering:\t'+file_name + '.avi')
    SaveAnimation('large.avi', view, ImageResolution=arg.size,
                  FrameRate=arg.fps, FrameWindow=[0, arg.n_frames-1])
    log.info('Compressing:\t'+file_name + '.avi')
    subprocess.call(['ffmpeg', '-i', 'large.avi', '-c:v', 'libx264', 
                     '-crf', '0', '-pix_fmt', 'yuv420p', '-nostdin', '-y',
                     '-loglevel', 'panic', file_name + '.avi'])
    if file_type == 'mp4':
        log.info('Converting:\t'+arg.output_filename)
        subprocess.call(['ffmpeg', '-i', file_name + '.avi', '-nostdin', '-y',
                         '-loglevel', 'panic', arg.output_filename])
        subprocess.call(['rm', '-i', '-f', file_name + '.avi'])                
    elif file_type == '':
        log.info('Converting:\t'+arg.output_filename)
        subprocess.call(['mkdir', arg.output_filename[:-1]])
        subprocess.call(['ffmpeg', '-i', file_name + '.avi', 
                         '-nostdin', '-y','-loglevel', 'panic',
                         arg.output_filename[:-1]+'/'+arg.output_filename[:-1]+'%04d.png'])
        subprocess.call(['rm', '-i', '-f', file_name + '.avi'])        
    subprocess.call(['rm', '-i', '-f', 'large.avi'])
    
elif file_type == 'png':
    log.info('Rendering:\t'+arg.output_filename)
    SaveScreenshot(arg.output_filename, view=view,
                   magnification=arg.magnification, quality=100)
else:
    log.info('Rendering:\t'+file_name + '.png')
    SaveAnimation('test/large.png', view, ImageResolution=arg.size,
                  FrameRate=arg.fps, FrameWindow=[0, arg.n_frames-1])

if arg.view:
    subprocess.call(['open', '-a', 'VLC.app', arg.output_filename])    

if arg.paraview:
    SaveState('state.pvsm')
    subprocess.call(['paraview', '--state='+os.getcwd()+'/state.pvsm'])
