#!/bin/sh
''':'
exec pvpython "$0"
'''

# state file generated using paraview version 5.4.0

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
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

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
line_pointsvtu = XMLUnstructuredGridReader(FileName=['/Users/Talon/GoogleDrive/projects/polaris/examples/line_points.vtu'])
line_pointsvtu.PointArrayStatus = ['uvw']

# create a new 'XML Unstructured Grid Reader'
line_pointsvtu_1 = XMLUnstructuredGridReader(FileName=['/Users/Talon/GoogleDrive/projects/polaris/examples/line_points.vtu'])
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
# renderView1.ViewSize = [1704, 706]

# get camera animation track for the view
cameraAnimationCue1 = GetCameraTrack(view=renderView1)

# create keyframes for this animation track

# create a key frame
keyFrame5995 = CameraKeyFrame()
keyFrame5995.Position = [-120.2020177082815, -122.67801984671057, 138.06875931583917]
keyFrame5995.FocalPoint = [43.49999999999992, 49.49999999999984, 19.50000000000001]
keyFrame5995.ViewUp = [0.37734093203346036, 0.2544268991803213, 0.8904385290325246]
keyFrame5995.ParallelScale = 68.72226713373185
keyFrame5995.PositionPathPoints = [0.0, 0.0, 10.0, 8.741676355490846, 0.0, 47.32730560840478, 43.25175634360137, 0.0, 64.02458169468926, 77.94587405466831, 0.0, 47.713148718459784, 87.10322551667325, 0.0, 10.485637873807402, 63.934923811705815, 0.0, -20.058992515100527, 25.617002971839423, 0.0, -21.276199152088722]
keyFrame5995.FocalPathPoints = [43.5, 49.5, 19.5]
keyFrame5995.ClosedPositionPath = 1

# create a key frame
keyFrame5996 = CameraKeyFrame()
keyFrame5996.KeyTime = 1.0
keyFrame5996.Position = [-120.2020177082815, -122.67801984671057, 138.06875931583917]
keyFrame5996.FocalPoint = [43.49999999999992, 49.49999999999984, 19.50000000000001]
keyFrame5996.ViewUp = [0.37734093203346036, 0.2544268991803213, 0.8904385290325246]
keyFrame5996.ParallelScale = 68.72226713373185

# initialize the animation track
cameraAnimationCue1.Mode = 'Path-based'
cameraAnimationCue1.KeyFrames = [keyFrame5995, keyFrame5996]

# get animation scene
animationScene1 = GetAnimationScene()

animationScene1.Play()

animationScene1.Play()

animationScene1.Play()

# Properties modified on animationScene1
animationScene1.NumberOfFrames = 300

animationScene1.Play()

# get camera animation track for the view
cameraAnimationCue1_1 = GetCameraTrack(view=renderView1)

# create keyframes for this animation track

# create a key frame
keyFrame6020 = CameraKeyFrame()
keyFrame6020.Position = [-96.82451378583984, -97.1026124261511, 69.28059239584957]
keyFrame6020.FocalPoint = [43.49999999999999, 49.50000000000004, 19.500000000000018]
keyFrame6020.ViewUp = [0.18940927480427416, 0.14829238777669196, 0.9706356135778278]
keyFrame6020.ParallelScale = 68.72226713373185
keyFrame6020.PositionPathPoints = [-96.8245, -97.1026, 69.2806, 71.51398105905348, -155.9379925701256, 45.4199505661183, 219.0840567904852, -62.47002173633574, 2.3433978675792773, 236.48329126454917, 114.00797074530075, -28.013858677186036, 110.81260161948899, 242.66239839197834, -23.146187098550897, -64.76088795643905, 228.11411534414222, 13.337719408952097, -160.07415254719407, 81.14862534392873, 54.390180450971954]
keyFrame6020.FocalPathPoints = [43.5, 49.5, 19.5]
keyFrame6020.ClosedPositionPath = 1

# create a key frame
keyFrame6021 = CameraKeyFrame()
keyFrame6021.KeyTime = 1.0
keyFrame6021.Position = [-96.82451378583984, -97.1026124261511, 69.28059239584957]
keyFrame6021.FocalPoint = [43.49999999999999, 49.50000000000004, 19.500000000000018]
keyFrame6021.ViewUp = [0.18940927480427416, 0.14829238777669196, 0.9706356135778278]
keyFrame6021.ParallelScale = 68.72226713373185

# initialize the animation track
cameraAnimationCue1_1.Mode = 'Path-based'
cameraAnimationCue1_1.KeyFrames = [keyFrame6020, keyFrame6021]

# get camera animation track for the view
cameraAnimationCue1_2 = GetCameraTrack(view=renderView1)

# create keyframes for this animation track

# create a key frame
keyFrame6023 = CameraKeyFrame()
keyFrame6023.Position = [-96.82451378583984, -97.1026124261511, 69.28059239584957]
keyFrame6023.FocalPoint = [43.49999999999999, 49.50000000000004, 19.500000000000018]
keyFrame6023.ViewUp = [0.18940927480427416, 0.14829238777669196, 0.9706356135778278]
keyFrame6023.ParallelScale = 68.72226713373185
keyFrame6023.PositionPathPoints = [-96.8245, -97.1026, 69.2806, 71.51398105905348, -155.9379925701256, 45.4199505661183, 219.0840567904852, -62.47002173633574, 2.3433978675792773, 236.48329126454917, 114.00797074530075, -28.013858677186036, 110.81260161948899, 242.66239839197834, -23.146187098550897, -64.76088795643905, 228.11411534414222, 13.337719408952097, -160.07415254719407, 81.14862534392873, 54.390180450971954]
keyFrame6023.FocalPathPoints = [43.5, 49.5, 19.5]
keyFrame6023.ClosedPositionPath = 1

# create a key frame
keyFrame6024 = CameraKeyFrame()
keyFrame6024.KeyTime = 1.0
keyFrame6024.Position = [-96.82451378583984, -97.1026124261511, 69.28059239584957]
keyFrame6024.FocalPoint = [43.49999999999999, 49.50000000000004, 19.500000000000018]
keyFrame6024.ViewUp = [0.18940927480427416, 0.14829238777669196, 0.9706356135778278]
keyFrame6024.ParallelScale = 68.72226713373185

# initialize the animation track
cameraAnimationCue1_2.Mode = 'Path-based'
cameraAnimationCue1_2.KeyFrames = [keyFrame6023, keyFrame6024]

animationScene1.Play()

# get camera animation track for the view
cameraAnimationCue1_3 = GetCameraTrack(view=renderView1)

# create keyframes for this animation track

# create a key frame
keyFrame6026 = CameraKeyFrame()
keyFrame6026.Position = [-96.82449999999999, -97.1026, 69.2806]
keyFrame6026.FocalPoint = [43.5, 49.5, 19.5]
keyFrame6026.ViewUp = [0.18940927480427416, 0.14829238777669196, 0.9706356135778278]
keyFrame6026.ParallelScale = 68.72226713373185
keyFrame6026.PositionPathPoints = [-96.8245, -97.1026, 69.2806, 71.51398105905348, -155.9379925701256, 45.4199505661183, 219.0840567904852, -62.47002173633574, 2.3433978675792773, 236.48329126454917, 114.00797074530075, -28.013858677186036, 110.81260161948899, 242.66239839197834, -23.146187098550897, -64.76088795643905, 228.11411534414222, 13.337719408952097, -160.07415254719407, 81.14862534392873, 54.390180450971954]
keyFrame6026.FocalPathPoints = [43.5, 49.5, 19.5]
keyFrame6026.ClosedPositionPath = 1

# create a key frame
keyFrame6027 = CameraKeyFrame()
keyFrame6027.KeyTime = 1.0
keyFrame6027.Position = [-96.82449999999999, -97.1026, 69.2806]
keyFrame6027.FocalPoint = [43.5, 49.5, 19.5]
keyFrame6027.ViewUp = [0.18940927480427416, 0.14829238777669196, 0.9706356135778278]
keyFrame6027.ParallelScale = 68.72226713373185

# initialize the animation track
cameraAnimationCue1_3.Mode = 'Path-based'
cameraAnimationCue1_3.KeyFrames = [keyFrame6026, keyFrame6027]

# current camera placement for renderView1
renderView1.CameraPosition = [-96.82449999999999, -97.1026, 69.2806]
renderView1.CameraFocalPoint = [43.5, 49.5, 19.5]
renderView1.CameraViewUp = [0.18940927480427416, 0.14829238777669196, 0.9706356135778278]
renderView1.CameraParallelScale = 68.72226713373185

# save animation
SaveAnimation('/Users/Talon/GoogleDrive/projects/polaris/examples/tester.avi', renderView1, ImageResolution=[1704, 706],
    FrameWindow=[0, 99])

# current camera placement for renderView1
renderView1.CameraPosition = [-96.82449999999999, -97.1026, 69.2806]
renderView1.CameraFocalPoint = [43.5, 49.5, 19.5]
renderView1.CameraViewUp = [0.18940927480427416, 0.14829238777669196, 0.9706356135778278]
renderView1.CameraParallelScale = 68.72226713373185

# save animation
SaveAnimation('/Users/Talon/GoogleDrive/projects/polaris/examples/tester2.avi', renderView1, ImageResolution=[1704, 706],
    FrameRate=10,
    FrameWindow=[0, 99])

# get camera animation track for the view
cameraAnimationCue1_4 = GetCameraTrack(view=renderView1)

# create keyframes for this animation track

# create a key frame
keyFrame6037 = CameraKeyFrame()
keyFrame6037.Position = [-95.71686108039174, -98.17259236414577, 69.22792795548338]
keyFrame6037.FocalPoint = [43.5, 49.49999999999999, 19.5]
keyFrame6037.ViewUp = [0.18940931284021836, 0.14829242751435934, 0.9706356000844472]
keyFrame6037.ParallelScale = 68.72226713373185
keyFrame6037.PositionPathPoints = [-95.71690000000001, -98.17259999999999, 69.2279, 73.01206888569389, -155.76811890013633, 45.10162549654929, 219.8620062324982, -61.18621571563145, 1.995415372509516, 235.96435734027435, 115.45394378112583, -28.133544147914108, 109.38149578540495, 243.19854909583086, -22.948871198509778, -66.04320719020684, 227.34295954138818, 13.70572913788314, -160.25703098816464, 79.64186258922804, 54.656031023427396]
keyFrame6037.FocalPathPoints = [43.5, 49.5, 19.5]
keyFrame6037.ClosedPositionPath = 1

# create a key frame
keyFrame6038 = CameraKeyFrame()
keyFrame6038.KeyTime = 1.0
keyFrame6038.Position = [-95.71686108039174, -98.17259236414577, 69.22792795548338]
keyFrame6038.FocalPoint = [43.5, 49.49999999999999, 19.5]
keyFrame6038.ViewUp = [0.18940931284021836, 0.14829242751435934, 0.9706356000844472]
keyFrame6038.ParallelScale = 68.72226713373185

# initialize the animation track
cameraAnimationCue1_4.Mode = 'Path-based'
cameraAnimationCue1_4.KeyFrames = [keyFrame6037, keyFrame6038]

# current camera placement for renderView1
renderView1.CameraPosition = [-95.71686108039174, -98.17259236414577, 69.22792795548338]
renderView1.CameraFocalPoint = [43.5, 49.49999999999999, 19.5]
renderView1.CameraViewUp = [0.18940931284021836, 0.14829242751435934, 0.9706356000844472]
renderView1.CameraParallelScale = 68.72226713373185

# save animation
SaveAnimation('/Users/Talon/GoogleDrive/projects/polaris/examples/tester5.avi', renderView1, ImageResolution=[1704, 706],
    FrameRate=30,
    FrameWindow=[0, 299])
