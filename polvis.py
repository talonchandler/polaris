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

parser.add_argument('--size', nargs=2, type=int, default=[1100, 700],
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
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1830, 678]
renderView1.AnnotationColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesVisibility = 0
renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
renderView1.CenterOfRotation = [43.5, 49.5, 19.5]
renderView1.StereoType = 0
renderView1.CameraPosition = [-92.8211, -99.84000000000006, 104.753]
renderView1.CameraFocalPoint = [43.5, 49.5, 19.5]
renderView1.CameraViewUp = [0.36295577383376965, 0.1895031731835262, 0.9123330825932512]
renderView1.CameraParallelScale = 68.7223
renderView1.Background = [1.0, 1.0, 1.0]

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView1.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.GridColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

# create a new 'XML Unstructured Grid Reader'
line_pointsvtu = XMLUnstructuredGridReader(FileName=[arg.input_filename])
line_pointsvtu.PointArrayStatus = ['uvw']

# create a new 'XML Unstructured Grid Reader'
line_pointsvtu_1 = XMLUnstructuredGridReader(FileName=[arg.input_filename])
line_pointsvtu_1.PointArrayStatus = ['uvw']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from line_pointsvtu
line_pointsvtuDisplay = Show(line_pointsvtu, renderView1)
# trace defaults for the display properties.
line_pointsvtuDisplay.Representation = '3D Glyphs'
line_pointsvtuDisplay.AmbientColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.ColorArrayName = [None, '']
line_pointsvtuDisplay.DiffuseColor = [0.6666666666666666, 0.0, 0.0]
line_pointsvtuDisplay.OSPRayScaleArray = 'uvw'
line_pointsvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
line_pointsvtuDisplay.Orient = 1
line_pointsvtuDisplay.SelectOrientationVectors = 'uvw'
line_pointsvtuDisplay.Scaling = 1
line_pointsvtuDisplay.ScaleFactor = 3.0
line_pointsvtuDisplay.SelectScaleArray = 'None'
line_pointsvtuDisplay.GlyphType = 'Arrow'
line_pointsvtuDisplay.GlyphTableIndexArray = 'None'
line_pointsvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
line_pointsvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
line_pointsvtuDisplay.ScalarOpacityUnitDistance = 12.143339580122927
line_pointsvtuDisplay.GaussianRadius = 4.95
line_pointsvtuDisplay.SetScaleArray = [None, '']
line_pointsvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
line_pointsvtuDisplay.OpacityArray = [None, '']
line_pointsvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'

# init the 'Arrow' selected for 'GlyphType'
line_pointsvtuDisplay.GlyphType.TipResolution = 1
line_pointsvtuDisplay.GlyphType.TipRadius = 0.0
line_pointsvtuDisplay.GlyphType.TipLength = 0.0
line_pointsvtuDisplay.GlyphType.ShaftResolution = 32
line_pointsvtuDisplay.GlyphType.ShaftRadius = 0.1

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
line_pointsvtuDisplay.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
line_pointsvtuDisplay.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
line_pointsvtuDisplay.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]

# show data from line_pointsvtu_1
line_pointsvtu_1Display = Show(line_pointsvtu_1, renderView1)
# trace defaults for the display properties.
line_pointsvtu_1Display.Representation = 'Outline'
line_pointsvtu_1Display.AmbientColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.ColorArrayName = [None, '']
line_pointsvtu_1Display.OSPRayScaleArray = 'uvw'
line_pointsvtu_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
line_pointsvtu_1Display.SelectOrientationVectors = 'None'
line_pointsvtu_1Display.ScaleFactor = 9.9
line_pointsvtu_1Display.SelectScaleArray = 'None'
line_pointsvtu_1Display.GlyphType = 'Arrow'
line_pointsvtu_1Display.GlyphTableIndexArray = 'None'
line_pointsvtu_1Display.DataAxesGrid = 'GridAxesRepresentation'
line_pointsvtu_1Display.PolarAxes = 'PolarAxesRepresentation'
line_pointsvtu_1Display.ScalarOpacityUnitDistance = 12.143339580122927
line_pointsvtu_1Display.GaussianRadius = 4.95
line_pointsvtu_1Display.SetScaleArray = [None, '']
line_pointsvtu_1Display.ScaleTransferFunction = 'PiecewiseFunction'
line_pointsvtu_1Display.OpacityArray = [None, '']
line_pointsvtu_1Display.OpacityTransferFunction = 'PiecewiseFunction'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
line_pointsvtu_1Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
line_pointsvtu_1Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
line_pointsvtu_1Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(line_pointsvtu)
# ----------------------------------------------------------------

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
renderView1.ViewSize = arg.size

# get animation scene
animationScene1 = GetAnimationScene()


# Properties modified on animationScene1
animationScene1.NumberOfFrames = arg.n_frames

# get camera animation track for the view
anim = GetCameraTrack(view=renderView1)

# create keyframes for orbit

# create a key frame
kf1 = CameraKeyFrame()
kf1.Position = [-95.71686108039174, -98.17259236414577, 69.22792795548338]
kf1.FocalPoint = [43.5, 49.5, 19.5]
kf1.ViewUp = [0.18940931284021836, 0.14829242751435934, 0.9706356000844472]
kf1.ParallelScale = 68.72226713373185
kf1.PositionPathPoints = [-95.71690000000001, -98.17259999999999, 69.2279, 73.01206888569389, -155.76811890013633, 45.10162549654929, 219.8620062324982, -61.18621571563145, 1.995415372509516, 235.96435734027435, 115.45394378112583, -28.133544147914108, 109.38149578540495, 243.19854909583086, -22.948871198509778, -66.04320719020684, 227.34295954138818, 13.70572913788314, -160.25703098816464, 79.64186258922804, 54.656031023427396]
kf1.FocalPathPoints = [43.5, 49.5, 19.5]
kf1.ClosedPositionPath = 1

# create a key frame
kf2 = CameraKeyFrame()
kf2.KeyTime = 1.0
kf2.Position = [-95.71686108039174, -98.17259236414577, 69.22792795548338]
kf2.FocalPoint = [43.5, 49.5, 19.5]
kf2.ViewUp = [0.18940931284021836, 0.14829242751435934, 0.9706356000844472]
kf2.ParallelScale = 68.72226713373185

# initialize the animation track
anim.Mode = 'Path-based'
anim.KeyFrames = [kf1, kf2]

# current camera placement for renderView1
renderView1.CameraPosition = [-95.71686108039174, -98.17259236414577, 69.22792795548338]
renderView1.CameraFocalPoint = [43.5, 49.5, 19.5]
renderView1.CameraViewUp = [0.18940931284021836, 0.14829242751435934, 0.9706356000844472]
renderView1.CameraParallelScale = 68.72226713373185

log.info('-----saving output-----')
file_name = arg.output_filename.split('.')[0]
file_type = arg.output_filename.split('.')[-1]

if file_type in ['avi', 'mp4', '']:
    log.info('Rendering:\t'+file_name + '.avi')
    SaveAnimation('large.avi', renderView1, ImageResolution=arg.size,
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
    SaveScreenshot(arg.output_filename, view=renderView1,
                   magnification=arg.magnification, quality=100)
else:
    log.info('Rendering:\t'+file_name + '.png')
    SaveAnimation('test/large.png', renderView1, ImageResolution=arg.size,
                  FrameRate=arg.fps, FrameWindow=[0, arg.n_frames-1])

if arg.view:
    subprocess.call(['open', '-a', 'VLC.app', arg.output_filename])    

if arg.paraview:
    SaveState('state.pvsm')
    subprocess.call(['paraview', '--state='+os.getcwd()+'/state.pvsm'])
