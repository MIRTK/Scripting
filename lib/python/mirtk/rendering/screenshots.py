##############################################################################
# Medical Image Registration ToolKit (MIRTK)
#
# Copyright 2016 Imperial College London
# Copyright 2016 Andreas Schuh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Python module for the rendering of screenshots."""

import os
import vtk


def iround(x):
    """Round floating point number and cast to int."""
    return int(round(x))


def nearest_voxel(index):
    """Get indices of nearest voxel."""
    i = iround(index[0])
    j = iround(index[1])
    k = iround(index[2])
    return (i, j, k)


def invert_matrix(m):
    """Get inverse of a vtkMatrix4x4."""
    inv = vtk.vtkMatrix4x4()
    inv.DeepCopy(m)
    inv.Invert()
    return inv


def index_to_point(index, origin, spacing):
    """Transform voxel indices to image data point coordinates."""
    x = origin[0] + index[0] * spacing[0]
    y = origin[1] + index[1] * spacing[1]
    z = origin[2] + index[2] * spacing[2]
    return (x, y, z)


def point_to_index(point, origin, spacing):
    """Transform image data point coordinates to voxel."""
    i = (point[0] - origin[0]) / spacing[0]
    j = (point[1] - origin[1]) / spacing[1]
    k = (point[2] - origin[2]) / spacing[2]
    return (i, j, k)


def matrix_to_affine(matrix):
    """Convert vtkMatrix4x4 to NiBabel 'affine' 2D array."""
    return [[matrix.GetElement(0, 0), matrix.GetElement(0, 1),
             matrix.GetElement(0, 2), matrix.GetElement(0, 3)],
            [matrix.GetElement(1, 0), matrix.GetElement(1, 1),
             matrix.GetElement(1, 2), matrix.GetElement(1, 3)],
            [matrix.GetElement(2, 0), matrix.GetElement(2, 1),
             matrix.GetElement(2, 2), matrix.GetElement(2, 3)],
            [matrix.GetElement(3, 0), matrix.GetElement(3, 1),
             matrix.GetElement(3, 2), matrix.GetElement(3, 3)]]


def range_to_level_window(min_value, max_value):
    """Convert min/max value range to level/window parameters."""
    window = max_value - min_value
    level = min_value + .5 * window
    return (level, window)


def auto_image_range(image, percentiles=(1, 99)):
    """Compute range for color transfer function."""
    stats = vtk.vtkImageHistogramStatistics()
    stats.SetInputData(image)
    stats.AutomaticBinningOn()
    stats.SetMaximumNumberOfBins(512)
    stats.SetAutoRangePercentiles(percentiles)
    stats.UpdateWholeExtent()
    return tuple(stats.GetAutoRange())


def auto_level_window(image, percentiles=(1, 99)):
    """Compute level/window for color transfer function."""
    return range_to_level_window(*auto_image_range(image, percentiles))


def add_contour(renderer, plane, polydata,
                transform=None, line_width=3, color=(1, 0, 0)):
    """Add contour of mesh cut by given image plane to render scene."""
    if transform:
        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetInputData(polydata)
        transformer.SetTransform(transform)
        transformer.Update()
        polydata = transformer.GetOutput()
    cutter = vtk.vtkCutter()
    cutter.SetInputData(polydata)
    cutter.SetCutFunction(plane)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cutter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.LightingOff()
    prop.SetRepresentationToWireframe()
    prop.SetLineWidth(line_width)

    if color:
        prop.SetColor(color)
        mapper.ScalarVisibilityOff()
    elif polydata.GetPointData().GetScalars():
        mapper.SetScalarModeToUsePointData()
        mapper.ScalarVisibilityOn()
    elif polydata.GetCellData().GetScalars():
        mapper.SetScalarModeToUseCellData()
        mapper.ScalarVisibilityOn()

    renderer.AddActor(actor)
    return actor


def slice_view(image, index, width, height, axis=2, polydata=[],
               transform=None, line_width=3, level_window=None):
    """Return vtkRenderer for orthogonal image slice."""

    # determine orientation of medical volume
    flip = [False, False, False]
    if transform:
        transform.Update()
        matrix = transform.GetMatrix()
        try:
            from nibabel import aff2axcodes
            codes = aff2axcodes(matrix_to_affine(matrix))
        except Exception:
            codes = ('L', 'A', 'S')
            if matrix.GetElement(0, 0) < 0:
                codes = 'R'
            if matrix.GetElement(1, 1) < 0:
                codes = 'P'
            if matrix.GetElement(2, 2) < 0:
                codes = 'I'
        if codes[0] == 'R':
            flip[0] = True
        if codes[1] == 'P':
            flip[1] = True
        if codes[2] == 'I':
            flip[2] = True

    dims = image.GetDimensions()
    if axis == 0:
        axes = (1, 2, 0)
        if width < 1:
            width = dims[1]
        if height < 1:
            height = dims[2]
        size = (1, width, height)
        up = (0, 0, 1)
    elif axis == 1:
        axes = (0, 2, 1)
        if width < 1:
            width = dims[0]
        if height < 1:
            height = dims[2]
        size = (width, 1, height)
        up = (0, 0, 1)
    elif axis == 2:
        axes = (0, 1, 2)
        if width < 1:
            width = dims[0]
        if height < 1:
            height = dims[1]
        size = (width, height, 1)
        up = (0, 1, 0)
    else:
        raise Exception("Invalid axis argument: {}".format(axis))

    spacing = image.GetSpacing()
    distance = 10. * spacing[axis]
    focal_point = index_to_point(index, image.GetOrigin(), spacing)
    position = list(focal_point)
    if flip[axis]:
        position[axis] = position[axis] - distance
    else:
        position[axis] = position[axis] + distance

    margin = 2
    extent = [index[0], index[0],
              index[1], index[1],
              index[2], index[2]]
    for i in range(3):
        if i != axis:
            radius = (size[i] - 1) / 2 + margin
            extent[2 * i] -= radius
            extent[2 * i + 1] += radius

    if flip[0] or flip[1] or flip[2]:
        flip_transform = vtk.vtkTransform()
        flip_transform.Translate(+focal_point[0], +focal_point[1], +focal_point[2])
        flip_transform.Scale(-1. if flip[0] else 1.,
                             -1. if flip[1] else 1.,
                             -1. if flip[2] else 1.)
        flip_transform.Translate(-focal_point[0], -focal_point[1], -focal_point[2])
        points_transform = vtk.vtkTransform()
        points_transform.SetMatrix(matrix)
        points_transform.PostMultiply()
        points_transform.Concatenate(flip_transform)
    else:
        flip_transform = None
        points_transform = None

    mapper = vtk.vtkImageSliceMapper()
    mapper.SetInputData(image)
    mapper.SetOrientation(axis)
    mapper.SetSliceNumber(extent[2 * axis])
    mapper.SetCroppingRegion(extent)
    mapper.CroppingOn()
    mapper.Update()

    actor = vtk.vtkImageSlice()
    actor.SetMapper(mapper)
    if flip_transform:
        actor.SetUserTransform(flip_transform)
    prop = actor.GetProperty()
    prop.SetInterpolationTypeToNearest()

    if not level_window:
        level_window = auto_level_window(image)
    prop.SetColorLevel(level_window[0])
    prop.SetColorWindow(level_window[1])

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    camera = renderer.GetActiveCamera()
    camera.SetViewUp(up)
    camera.SetPosition(position)
    camera.SetFocalPoint(focal_point)
    camera.SetParallelScale(.5 * max((size[axes[0]] - 1) * spacing[axes[0]],
                                     (size[axes[1]] - 1) * spacing[axes[1]]))
    camera.SetClippingRange(distance - .5 * spacing[axis],
                            distance + .5 * spacing[axis])
    camera.ParallelProjectionOn()

    # add contours of polygonal data intersected by slice plane
    colors = [
        (1, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1)
    ]
    if isinstance(polydata, vtk.vtkPolyData):
        polydata = [polydata]
    if len(polydata) > len(colors):
        raise Exception('Maximum number of contour overlays is: {}'.format(len(colors)))
    for i in xrange(len(polydata)):
        add_contour(renderer, plane=mapper.GetSlicePlane(),
                    polydata=polydata[i], transform=points_transform,
                    line_width=line_width, color=colors[i])
    return renderer


def take_screenshot(window, path=None):
    """Takes vtkRenderWindow instance and writes a screenshot of the rendering.

    window : vtkRenderWindow
        The render window from which a screenshot is taken.
    path : str
        File name path of output PNG file.
        A .png file name extension is appended if missing.

    """
    _offscreen = window.GetOffScreenRendering()
    window.OffScreenRenderingOn()
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(window)
    window_to_image.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(window_to_image.GetOutputPort())
    if path:
        if os.path.splitext(path)[1].lower() != '.png':
            path += '.png'
        writer.SetFileName(path)
    else:
        writer.WriteToMemoryOn()
    writer.Write()
    window.SetOffScreenRendering(_offscreen)
    if writer.GetWriteToMemory():
        from IPython.display import Image
        data = str(buffer(writer.GetResult()))
        return Image(data)


def take_orthogonal_screenshots(image, center, length=0, size=(512, 512),
                                prefix='screenshot', delim='_',
                                level_window=None, qform=None, polydata=[]):
    """Take three orthogonal screenshots of the given image patch.

    Arguments
    ---------

    image : vtkImageData
        Volume data.
    center : tuple
        Coordinates of patch center point.
    length : float
        Side length of patch in mm.
    size : list, tuple, int
        Either int or 2-tuple/-list of screenshot image size.
    prefix : str
        Common output path prefix of screenshot files.
    delim : str
        Delimiting string between file name prefix,
        patch center index, and orthogonal view suffix.
    qform : vtkMatrix4x4, optional
        Homogeneous image world to vtkImageData coordinates transformation.
    level_window : tuple, list, optional
        2-tuple/-list of level and window color transfer function parameters.
        When not specified, the auto_level_window function with default percentiles
        is used to compute an intensity range that is robust to outliers.
    polydata : vtkPolyData, list, optional
        List of vtkPolyData objects to be cut by each orthogonal
        image slice plane and the contours rendered over the image.
        When a `qform` matrix is given, the points are transformed
        to image coordinates using the inverse of the `qform` matrix.

    Returns
    -------

    Nothing

    """
    if isinstance(size, int):
        size = (size, size)
    if not level_window:
        level_window = auto_level_window(image)
    spacing = image.GetSpacing()
    index = nearest_voxel(point_to_index(center, image.GetOrigin(), spacing))
    if qform:
        linear_transform = vtk.vtkMatrixToLinearTransform()
        linear_transform.SetInput(invert_matrix(qform))
        linear_transform.Update()
    else:
        linear_transform = None
    args = dict(
        polydata=polydata,
        transform=linear_transform,
        level_window=level_window
    )
    infix = ','.join(['{:03d}'.format(idx) for idx in index])
    suffix = ('3-sagittal', '2-coronal', '1-axial')
    directory = os.path.dirname(prefix)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    for i in xrange(3):
        j = (i + 1) % 3
        axis = (i + 2) % 3
        width = iround(length / spacing[i])
        height = iround(length / spacing[j])
        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1
        renderer = slice_view(image, axis=axis, index=index, width=width, height=height, **args)
        window = vtk.vtkRenderWindow()
        window.SetSize(size)
        window.AddRenderer(renderer)
        take_screenshot(window, path=delim.join([prefix, infix, suffix[axis]]))
