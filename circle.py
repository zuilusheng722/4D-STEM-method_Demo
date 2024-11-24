import os
import warnings
from math import pi
import numpy as np
import cv2
import matplotlib.colors as col
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits import axes_grid1
from natsort import natsorted
from scipy import signal
from scipy import stats, optimize
from scipy.optimize import minimize
from skimage.filters import threshold_otsu
from scipy.ndimage import rotate
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from scipy.spatial import ConvexHull
from scipy.stats import zscore
from scipy import stats
from scipy import ndimage
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
# Ignore all warnings
warnings.filterwarnings("ignore")

def readData(dname):
    """
    Read in a 4D-STEM data file.

    Parameters
    ----------
    dname : str
        Name of the data file.

    Returns
    -------
    data: 4D array of int or float
        The read-in 4D-STEM data.

    """
    dimy = 130
    dimx = 128

    file = open(dname, 'rb')
    data = np.fromfile(file, np.float32)
    pro_dim = int(np.sqrt(len(data) / dimx / dimy))

    data = np.reshape(data, (pro_dim, pro_dim, dimy, dimx))
    data = data[:, :, 1:dimx + 1, :]

    file.close()
    return data

def handData(data):
    """
    Process the data array by replacing NaN values with 0, shifting the data values to non-negative,
    and adding a small positive offset.

    Args:
    data: Input data array

    Returns:
    Processed data array
    """
    # Replace NaN values in the data array with 0
    data[np.where(np.isnan(data) == True)] = 0

    # Shift the data values to non-negative if there are negative values
    # and add a small positive offset to avoid division by zero in subsequent calculations
    data -= data.min() if data.min() < 0 else 0
    data += 10 ** (-17)

    return data
    return data
  
def correction(array):
    """
    Apply phase correction to the complex array.

    Args:
    array: Input complex array

    Returns:
    corrected_abs: Absolute values of the phase-corrected array
    corrected_angle: Phase angles of the phase-corrected array
    """
    corrected = array * np.exp(1j * np.angle(array - np.mean(array)))
    corrected_abs = np.abs(corrected)
    corrected_angle = np.angle(corrected)
    return corrected_abs, corrected_angle

def filtering(data):
    """
    Apply a series of filtering operations to the input data.

    Args:
    data: Input data array

    Returns:
    binary: Filtered binary image
    """
    # Apply median blur with kernel size 5 to reduce noise
    img_median = cv2.medianBlur(data, 5)

    # Apply Gaussian blur with kernel size (5, 5) to further smooth the image
    gaussianBlur = cv2.GaussianBlur(img_median, (5, 5), 0)
    
    # Normalization
    gaussianBlur = (((gaussianBlur - gaussianBlur.min()) / (gaussianBlur.max() - gaussianBlur.min())) * 255).astype(np.uint8)

    # Apply Otsu's thresholding to convert the image to binary
    ret, binary = cv2.threshold(gaussianBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary
    
def calculate_centroid1(data):
    """
    Calculate the centroid coordinates of the input data.

    Args:
    data: Input data array

    Returns:
    x: X-coordinate of the centroid
    y: Y-coordinate of the centroid
    """
    # Apply filtering operations to the data
    data = filtering(data)

    # Perform phase correction on the filtered data
    magnitude, phase = correction(data)
    magnitude = magnitude * np.exp(1j * phase)

    # Create a grid of indices
    y, x = np.mgrid[:magnitude.shape[0], :magnitude.shape[1]]

    # Calculate the total mass (sum of magnitude values)
    total_mass = np.sum(magnitude)

    # Calculate the centroid coordinates
    y = np.sum(y * magnitude) / total_mass
    x = np.sum(x * magnitude) / total_mass

    # Return the real values of x and y coordinates
    return x.real, y.real

def calculate_centroid(data,flag=False):
    """  
    Calculate the centroid coordinates of the input data using a weighted average method.  
  
    Args:  
        data (np.ndarray): Input data array, expected to be a 2D NumPy array.  
        flag (bool, optional): If True, apply filtering to the data before calculating the centroid.  
            Defaults to False.  
  
    Returns:  
        tuple: A tuple containing the X-coordinate (cx) and Y-coordinate (cy) of the centroid.  
    """ 
    if flag == True:
        # Filtering
        data = filtering(data)
    # Create arrays of x and y indices
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])

    # Create a meshgrid of x and y indices
    X, Y = np.meshgrid(x, y)

    # Calculate the centroid coordinates
    cx = np.sum(X * data) / np.sum(data)
    cy = np.sum(Y * data) / np.sum(data)

    # Return the centroid coordinates
    return cx, cy

def zy_zh(matrix):
    # Function to reverse the order of rows in a matrix
    # Get the length of the matrix
    length = len(matrix)   
    # Iterate through half of the rows of the matrix
    for j in range(length // 2):
        # Iterate through all columns of the matrix
        for i in range(length):
            # Swap elements between the current row and its corresponding row from the bottom
            matrix[j][i], matrix[length - j - 1][i] = matrix[length - j - 1][i], matrix[j][i]
    # Return the transformed matrix
    return matrix

def handle_missing_values(x):
    """
    Replace missing values in an array using linear interpolation.

    Args:
    x (array_like): Input array with missing values.

    Returns:
    array_like: Array with missing values replaced using linear interpolation.
    """
    x_mean = np.nanmean(x)
    # Check if each element is a missing value
    is_missing = np.logical_or(x == 0, np.isnan(x))

    # Replace missing values with nearby points using linear interpolation
    for i in range(x.shape[0]):
        if is_missing[i] or abs(x[i] - x_mean) > 10:
            # Find nearby points
            neighbors = []
            for ii in range(-1, 2):
                if i + ii >= 0 and i + ii < x.shape[0] and not is_missing[i + ii] and abs(x[i + ii] - x_mean) <= 10:
                    neighbors.append(x[i + ii])
            # If there are valid nearby points, perform linear interpolation
            if len(neighbors) > 0:
                x[i] = np.mean(neighbors)
            # If there are no valid nearby points, set the value to the mean
            else:
                x[i] = x_mean

    return x

def detect_and_smooth_jumps(x, y):
    """
    Detect and smooth jump points and local extrema.

    Args:
    x: One-dimensional array representing the x-coordinates.
    y: One-dimensional array representing the y-coordinates.

    Returns:
    tuple: A tuple containing the smoothed x-coordinate array and y-coordinate array.
    """
    window_size = 3

    # Detect jump points
    diff_x = np.diff(x)
    jump_indices = np.where(np.abs(x[2:-2] - x[1:-3]) > 0.01 * np.mean(np.abs(diff_x)))[0] + 2

    # Smooth jump points
    for j in jump_indices:
        if j >= window_size and j < len(x) - window_size:
            x[j] = np.mean(x[j - window_size:j + window_size + 1])
            y[j] = np.mean(y[j - window_size:j + window_size + 1])

    # Detect local extrema
    extrema_indices = []
    for i in range(window_size, len(x) - window_size):
        if y[i] == np.max(y[i - window_size:i + window_size + 1]) or y[i] == np.min(
                y[i - window_size:i + window_size + 1]):
            threshold = threshold_otsu(y[i - window_size:i + window_size + 1])
            if y[i] > threshold:
                extrema_indices.append(i)

    # Smooth local extrema
    for i in extrema_indices:
        if i >= window_size and i < len(x) - window_size:
            x[i] = np.mean(x[i - window_size:i + window_size + 1])
            y[i] = np.mean(y[i - window_size:i + window_size + 1])

    return x, y

def handle_missing_values(x):
    """
    Replace missing values in an array using linear interpolation.

    Args:
    x (array_like): Input array with missing values.

    Returns:
    array_like: Array with missing values replaced using linear interpolation.
    """
    x_mean = np.nanmean(x)
    # Check if each element is a missing value
    is_missing = np.logical_or(x == 0, np.isnan(x))

    # Replace missing values with nearby points using linear interpolation
    for i in range(x.shape[0]):
        if is_missing[i] or abs(x[i] - x_mean) > 10:
            # Find nearby points
            neighbors = []
            for ii in range(-1, 2):
                if i + ii >= 0 and i + ii < x.shape[0] and not is_missing[i + ii] and abs(x[i + ii] - x_mean) <= 10:
                    neighbors.append(x[i + ii])
            # If there are valid nearby points, perform linear interpolation
            if len(neighbors) > 0:
                x[i] = np.mean(neighbors)
            # If there are no valid nearby points, set the value to the mean
            else:
                x[i] = x_mean

    return x

def Handling_missing_values(x):
    # Calculate the mean value of x excluding NaNs
    x_ = np.nanmean(x)
    
    # Determine if each element is a missing value
    is_missing = np.logical_or(x == 0, np.isnan(x))

    # Replace missing values with neighboring points using linear interpolation
    for i in range(x.shape[0]):
        if is_missing[i] or abs(x[i] - x_) > 10:
            # Find neighboring points
            neighbors = []
            for ii in range(-1, 2):
                if i + ii >= 0 and i + ii < x.shape[0] and not is_missing[i + ii] and abs(x[i + ii] - x_) <= 10:
                    neighbors.append(x[i + ii])
            # If there are valid points nearby, perform linear interpolation
            if len(neighbors) > 0:
                x[i] = np.mean(neighbors)
            # If there are no valid points nearby, set the point to the mean
            else:
                x[i] = x_

    return x
    
def handle_X_Y(x, y):
    """
    Preprocesses x and y arrays by handling missing values, subtracting mean, and computing Z values.

    Args:
    x: One-dimensional array representing the x-coordinates.
    y: One-dimensional array representing the y-coordinates.

    Returns:
    tuple: A tuple containing Z values, preprocessed x array, and preprocessed y array.
    """
    # Convert input arrays to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Handle missing values in x and y arrays
    x = Handling_missing_values(x)
    y = Handling_missing_values(y)

    # Subtract mean from x and y arrays
    x = np.mean(x) - np.array(x)
    y = np.mean(y) - np.array(y)

    #x, y = detect_and_smooth_jumps(x, y)
    # Compute Z values using Euclidean distance
    Z = np.hypot(x, y)

    return Z, x, y  

def add_colorbar(im, aspect=20, pad_fraction=0.5, a=30, label="", format=None, **kwargs):
    """Add a vertical color bar to an image plot with a unit label.

    Args:
    im: The image plot to which the color bar will be added.
    aspect (float, optional): The aspect ratio of the color bar. Defaults to 20.
    pad_fraction (float, optional): The fraction of the original axes to use for padding. Defaults to 0.5.
    a (int, optional): The font size for label and ticks on the color bar. Defaults to 30.
    label (str, optional): The label for the color bar. Defaults to "".
    format (str, optional): The format for the color bar ticks. Defaults to None.
    **kwargs: Additional keyword arguments to be passed to the color bar.

    Returns:
    matplotlib.colorbar.Colorbar: The color bar object.
    """
    # Divide the color bar location
    divider = make_axes_locatable(im.axes)
    width = axes_size.AxesY(im.axes, aspect=1. / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)

    # Add label and ticks to the color bar
    cb = im.axes.figure.colorbar(im, cax=cax, format=format, **kwargs)
    cb.ax.tick_params(labelsize=a)  # Set tick label size
    cb.set_label(label, fontsize=a, rotation=270, labelpad=a)  # Set color bar label and label font size
    return cb

def readCsv(path):
    """
    Read data from a CSV file.

    Args:
    path (str): The path to the CSV file.

    Returns:
    tuple: A tuple containing x and y data arrays.
    """
    # Read the CSV file
    df = pd.read_csv(path, dtype='float32')
    x = df["x"].values.tolist()
    y = df["y"].values.tolist()
    x = np.array(x)
    y = np.array(y)
    return x, y

def cut_data(x, y, a, b, c, d):
    """
    Cut the data arrays x and y.

    Args:
    x (array_like): The x data array.
    y (array_like): The y data array.
    a (int): The start index for rows.
    b (int): The end index for rows.
    c (int): The start index for columns.
    d (int): The end index for columns.

    Returns:
    tuple: A tuple containing cut x and y data arrays.
    """
    i = int(np.sqrt(len(x)))
    x = x.reshape(i, i)[a:b, c:d].reshape(-1)
    y = y.reshape(i, i)[a:b, c:d].reshape(-1)
    return x, y

def field(x, y, scan_rotation):
    """
    Compute the magnitude and angle of a vector field.

    Args:
    x (array_like): The x components of the vectors.
    y (array_like): The y components of the vectors.
    scan_rotation (float): The rotation angle of the scan.

    Returns:
    tuple: A tuple containing the magnitude and angle of the vector field.
    """
    # Compute the magnitude of the vectors
    mag = np.sqrt(x ** 2 + y ** 2)
    # Compute the angle of the vectors
    angle = np.arctan2(y, x)
    # Adjust the angle by the scan rotation
    angle = angle + scan_rotation
    return mag, angle
 
def create_hsv_color_wheel(size=256, center_black=True):
    """
    Create a HSV color wheel image.

    Args:
    size (int, optional): The size of the color wheel image. Defaults to 256.
    center_black (bool, optional): Whether to center black in the color wheel. Defaults to True.

    Returns:
    array_like: An array representing the HSV color wheel image.
    """
    # Create meshgrid for x and y coordinates
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    r = np.sqrt(x ** 2 + y ** 2)  # Compute distance from center
    theta = np.arctan2(y, x)  # Compute angle from x-axis
    hsv = np.zeros((size, size, 3), dtype=np.float32)

    # Hue: Adjust angle to [0, 1] range and assign to hue channel
    hsv[..., 0] = ((theta + np.pi / 4) % (2 * np.pi)) / (2 * np.pi)

    # Saturation: Assign distance from center to saturation channel
    hsv[..., 1] = np.clip(r, 0, 1)

    # Value: Assign value channel based on center_black parameter
    if center_black:
        hsv[..., 2] = hsv[..., 1]  # Set value to saturation
    else:
        hsv[..., 2] = 1.0  # Set value to maximum

    # Convert HSV to RGB
    return col.hsv_to_rgb(hsv)

def smooth_matrix(matrix):
    """
    Smooth the input matrix.

    Args:
    matrix (array_like): The input matrix to be smoothed.

    Returns:
    array_like: The smoothed matrix.
    """
    # Compute row and column means
    row_mean = np.mean(matrix, axis=1)
    col_mean = np.mean(matrix, axis=0)

    # Smooth each row and column
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs(matrix[i, j] - row_mean[i]) > 5 * np.std(matrix[i, :]):
                matrix[i, j] = row_mean[i]
            if abs(matrix[i, j] - col_mean[j]) > 5 * np.std(matrix[:, j]):
                matrix[i, j] = col_mean[j]

    # Define the smoothing filter kernel
    kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9]])

    # Smooth the matrix using convolution
    smoothed_matrix = np.zeros_like(matrix)
    for i in range(smoothed_matrix.shape[0]):
        for j in range(smoothed_matrix.shape[1]):
            sub_matrix = matrix[max(0, i - 1):min(matrix.shape[0], i + 2), max(0, j - 1):min(matrix.shape[1], j + 2)]
            if sub_matrix.shape != (3, 3):
                pad_width = ((0, 3 - sub_matrix.shape[0]), (0, 3 - sub_matrix.shape[1]))
                sub_matrix = np.pad(sub_matrix, pad_width=pad_width, mode='constant')
            smoothed_matrix[i, j] = np.sum(sub_matrix * kernel)

    return smoothed_matrix

def add_color_wheel_to_image(axs, scan_rotation=0, angles=-135):
    """
    Add a color wheel to an image plot.

    Args:
    axs: The axes object to which the color wheel will be added.
    scan_rotation (float, optional): The rotation angle of the scan. Defaults to 0.
    angles (float, optional): The angle for rotating the color wheel. Defaults to -135.

    Returns:
    None
    """
    size = 256
    color_wheel = create_hsv_color_wheel(size=360)

    # Define center coordinates and radius
    center = (180, 180)
    radius = 180

    # Generate 2D coordinate grid
    x = np.arange(360) - center[0]
    y = np.arange(360) - center[1]
    xx, yy = np.meshgrid(x, y)

    # Calculate circle equation and set values inside circle to 1
    mask = (xx ** 2 + yy ** 2) >= (radius - 10) ** 2
    alpha = np.ones_like(color_wheel[..., 0])
    alpha[mask] = 0
    color_wheel = np.dstack((color_wheel, alpha))

    # Convert to HSV color space
    color_rgb = color_wheel[..., :3]  # Extract RGB channels
    color_hsv = col.rgb_to_hsv(color_rgb)  # Convert to HSV color space

    # Convert back to RGB color space
    color_rgb = col.hsv_to_rgb(color_hsv)
    color_wheel = np.dstack((color_rgb, alpha))

    # Rotate the color wheel
    color_wheel = rotate(color_wheel, angles, reshape=False)

    # Flip the color wheel along the middle horizontal axis
    color_wheel = np.flipud(color_wheel)

    # Rotate the color wheel according to scan rotation
    degrees = np.rad2deg(scan_rotation)
    color_wheel = rotate(color_wheel, 2 * degrees, reshape=False)

    # Plot the circular colormap
    axs.imshow(color_wheel, interpolation='nearest')

    axs.axis('off')
    
def drawing(axs, x, y, scan_rotation, angles=-135, step=2):
    """
    Draw a colored plot with vectors representing a vector field.

    Args:
    axs: The axes object to which the plot will be drawn.
    x (array_like): The x components of the vectors.
    y (array_like): The y components of the vectors.
    scan_rotation (float): The rotation angle of the scan.
    angles (float, optional): The angle for rotating the color wheel. Defaults to -135.
    step (int, optional): The sampling step for vectors. Defaults to 2.

    Returns:
    None
    """
    a = int(np.sqrt(len(x)))  # Determine the size of the grid
    U = x.reshape(a, a)  # Reshape the x components into a grid
    V = y.reshape(a, a)  # Reshape the y components into a grid

    # Calculate the magnitude and angle of the vector field
    mag, angle = field(U, V, scan_rotation)
    #mag = smooth_matrix(mag)  # Smooth the magnitude

    # Calculate hue and brightness for coloring
    hue = angle / (2 * np.pi) % 1  
    brightness = mag / np.max(mag)  

    # Calculate color mapping
    cmap = 'hsv'
    norm = col.Normalize(vmin=np.min(brightness), vmax=np.max(brightness))
    colors = col.hsv_to_rgb(np.dstack((hue, np.ones_like(hue), brightness)))

    # Draw the colored heatmap
    X, Y = np.meshgrid(np.arange(a), np.arange(a))
    axs.imshow(colors, interpolation='bicubic', origin='lower', cmap=cmap, norm=norm)#, vmin=0, vmax=1)

    # Rotate the arrow directions
    rot_matrix = np.array([[np.cos(scan_rotation), -np.sin(scan_rotation)], [np.sin(scan_rotation), np.cos(scan_rotation)]])
    directions = np.dot(rot_matrix, np.vstack([x, y]))
    u_rotated, v_rotated = directions[0], directions[1]
    X = np.array(X)
    Y = np.array(Y)
    u = np.array(u_rotated.reshape(a, a))
    v = np.array(v_rotated.reshape(a, a))

    # img the original data
    step = step
    X_sampled = X[::step, ::step]
    Y_sampled = Y[::step, ::step]
    u_sampled = u[::step, ::step]
    v_sampled = v[::step, ::step]

    # Draw the vector arrows
    axs.quiver(X_sampled, Y_sampled, u_sampled, v_sampled,
               angles='xy', color="w", pivot='mid',
               units="width", scale_units='width',
               minshaft=1.5, width=0.005)

    # Draw the circular color wheel
    circle_axs = axs.inset_axes([0.85, 0.85, 0.15, 0.15])
    add_color_wheel_to_image(circle_axs, scan_rotation, angles=angles)
    #axs.axis('off')
    axs.set_aspect('equal')  # Set equal aspect ratio for better visualization

def getAngle(img):
    """
    Compute the gradient magnitude and angle of an image using Sobel filters.

    Args:
    img (numpy.ndarray): The input image.

    Returns:
    tuple: A tuple containing the gradient magnitude image and the gradient angle image.
    """
    # Sobel filters for computing gradients
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Add padding to the image
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # Compute gradients in x and y directions
    img_x = cv2.filter2D(img.astype(np.float32), -1, sobelX, borderType=cv2.BORDER_CONSTANT)
    img_y = cv2.filter2D(img.astype(np.float32), -1, sobelY, borderType=cv2.BORDER_CONSTANT)

    # Compute gradient magnitude and angle
    gradient_img, angle = cv2.cartToPolar(img_x.astype(np.float32), img_y.astype(np.float32))

    return gradient_img, angle

def getX(img):
    """
    Extract non-zero pixel coordinates from the input image.

    Args:
    img (numpy.ndarray): The input image.

    Returns:
    list: A list containing the coordinates of non-zero pixels.
    """
    pxh, pxw = img.shape
    x = []
    for i in range(pxh):
        for j in range(pxw):
            if img[i][j] != 0:
                img[i][j] = 1
                x.append([i, j])
    return x

def r(x, y, xc, yc):
    """
    Compute the distance between points (x, y) and the center (xc, yc).

    Args:
    x (array_like): x coordinates of the points.
    y (array_like): y coordinates of the points.
    xc (float): x coordinate of the center.
    yc (float): y coordinate of the center.

    Returns:
    array_like: The distances between the points and the center.
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    
def generateAvg(data):
    """
    Generate an average (sum) pattern from the 4D dataset.

    Parameters
    ----------
    data : 2D array of int or float
        Array of the 2D dataset.

    Returns
    -------
    avg_pat: 2D array of int or float
        An average (sum) difffraction pattern.

    """
    pro_y, pro_x = data.shape
    avg_pat = data * 1
    avg_pat[:, :] = 0
    for row in range(pro_y):
        for col in range(pro_x):
            avg_pat += data[row, col]

    return avg_pat

def least_squares_circle(coords):
    """
    Fit a circle to a set of coordinates using the least squares method.

    Args:
    coords (numpy.ndarray): An array of shape (n, 2) containing the coordinates of points.

    Returns:
    tuple: A tuple containing the center coordinates (xc, yc), the radius (R), and the residual.
    """
    x = coords[:, 0]  # Extract x coordinates from the input array
    y = coords[:, 1]  # Extract y coordinates from the input array

    center_estimate = np.mean(coords, axis=0)  # Estimate the center of the circle

    bounds = [(None, None), (None, None)]  # Define bounds for optimization (no bounds on xc and yc)

    # Optimize the center using least squares with the trust-constr method
    result = minimize(f, center_estimate, args=(x, y), method='trust-constr', bounds=bounds)
    center = result.x  # Extract the optimized center
    xc, yc = center  # Separate x and y coordinates of the center

    Ri = r(x, y, *center)  # Compute distances between points and the center
    R = Ri.mean()  # Compute the mean radius
    residu = np.sum((Ri - R) ** 2)  # Compute the residual

    return xc, yc, R, residu

def f(c, x, y):
    """
    Objective function for least squares circle fitting.

    Args:
    c (tuple): A tuple containing the center coordinates (xc, yc) of the circle.
    x (array_like): The x coordinates of the points.
    y (array_like): The y coordinates of the points.

    Returns:
    float: The sum of squared residuals.
    """
    Ri = r(x, y, *c)  # Compute distances between points and the center
    return np.sum(np.square(Ri - Ri.mean()))  # Compute the sum of squared residuals

def Fitting_circle(data):
    """
    Fit a circle to a binary image.

    Args:
    data (numpy.ndarray): The binary image data.

    Returns:
    tuple: A tuple containing the center coordinates (xc, yc), the radius (r), and the residual.
    """
    # Filtering
    binary = filtering(data)

    # Compute gradient magnitude and angle
    grandien_img, angle = getAngle(binary)

    # Extract non-zero pixel coordinates
    x_center = getX(grandien_img)
    a = np.array(x_center)

    # If no coordinates are found, return default values
    if len(a) == 0:
        xc = 0
        yc = 0
        r = 0
        residual = []
    else:
        # Fit a circle using least squares
        xc, yc, r, residual = least_squares_circle(a)
        return xc, yc, r

def generate_ring_kernel(radius, ring_size):
    """
    Generate a ring-shaped convolution kernel with a given radius and ring size.

    Parameters
    ----------
    radius : int
        The radius of the kernel.
    ring_size : int
        The size of the ring.

    Returns
    -------
    kernel : ndarray
        The generated convolution kernel.
    """
    # Define the size of the kernel
    kernel_size = 2 * radius + 1
    # Generate a convolution kernel initialized with zeros
    kernel = np.zeros((kernel_size, kernel_size))
    # Use meshgrid to generate 2D coordinate matrices
    x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
    # Calculate the coordinates of the center point
    center = (radius, radius)
    # Calculate the distance from each pixel to the center point
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    # Set the values of pixels between radius and radius - ring_size to 1
    kernel[(distance <= radius) & (distance > radius - ring_size)] = 1
    # Normalize the kernel by dividing all values by the sum of the kernel
    kernel /= np.sum(kernel)
    # Return the generated convolution kernel
    return kernel

def get_Angle(img):
    """ Calculate Sobel gradient map and edge angle """
    # Use cv2.Sobel optimization
    img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude, angle = cv2.cartToPolar(img_x, img_y)
    return grad_magnitude, angle


def getEdgePoints(img):
    """ Get the coordinates of the edge points """
    pxh, pxw = img.shape
    edge_points = []
    for i in range(pxh):
        for j in range(pxw):
            if img[i, j] != 0:
                edge_points.append([i, j])
    return np.array(edge_points)


def circle_residual(c, x, y):
    """ Calculate the residuals of the fitted circle """
    Ri = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2)
    return np.sum((Ri - Ri.mean()) ** 2)


def fit_circles(coords):
    """ Fit a circle using the least squares method """
    x = coords[:, 0]
    y = coords[:, 1]

    # Preliminary estimate of the center position
    center_estimate = np.mean(coords, axis=0)

    # Least squares optimization using the trust-constr method
    result = minimize(circle_residual, center_estimate, args=(x, y), method='trust-constr')
    xc, yc = result.x

    # Calculate the fitting radius
    Ri = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    R = Ri.mean()

    return xc, yc, R


# Edge Detection
def edge_detection_fitting_circles(image,flag=False):
    if flag==True:
        # Median filter and Gaussian filter preprocessing
        image = cv2.medianBlur(image, 5)
        image = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get edge graph
    grad_magnitude, _ = get_Angle(binary)

    # Get valid edge points
    edge_points = getEdgePoints(grad_magnitude)
    if len(edge_points) < 5:  # Too few edge points to fit a circle
        return 0, 0, 0

    # Fitting circle
    yc, xc, R = fit_circles(edge_points)

    return xc, yc, R

def getCorre(data1, data2, step=5, flag=False):
    """
    Calculate cross-correlation.

    Parameters
    ----------
    data1 : 2D array of int or float
        The first 2D array.
    data2 : 2D array of int or float
        The second 2D array.
    step : int, optional
        The degree of the polynomial used for fitting. Default is 5.
    flag : bool, optional
        Flag indicating whether to perform filtering. Default is False.

    Returns
    -------
    tuple
        Coordinates (x, y) of the cross-correlation peak.
    """
    pxh, pxw = data1.shape

    if flag == True:
        # Filtering
        data1 = filtering(data1)

    correlation = signal.correlate2d(data1, data2, boundary="symm", mode='same')  # Image registration
    #correlation = np.log10(correlation + 1)  # Log transformation
    horizon = np.sum(correlation, axis=1)  # Sum along the horizontal axis
    vertical = np.sum(correlation, axis=0)  # Sum along the vertical axis
    # Polynomial Fitting
    try:
        z1 = np.polyfit(np.arange(0, pxh), horizon, step)  # Fit a polynomial of degree 'step' to horizontal sums
        z2 = np.polyfit(np.arange(0, pxh), vertical, step)  # Fit a polynomial of degree 'step' to vertical sums
        p1 = np.poly1d(z1)  # Polynomial coefficients
        p2 = np.poly1d(z2)  # Polynomial coefficients
        yyyd1 = np.polyder(p1, 1)
        yyyd2 = np.polyder(p2, 1)  # First derivative of the polynomial

        if yyyd1.r.any():  # Check if the list is empty
            x = min(yyyd1.r, key=lambda x: abs(x - pxh / 2)).real# Get the nearest extremum point to the middle of the image
        else:
            x = 0

        if yyyd2.r.any():  # Check if the list is empty
            y = min(yyyd2.r, key=lambda x: abs(x - pxh / 2)).real# Get the nearest extremum point to the middle of the image
        else:
            y = 0

    except np.linalg.LinAlgError:  # Catching exceptions during polynomial fitting
        x = 0
        y = 0
    return y, x

    
def find_circle_center(img, r, theta_step=0.001, threshold_factor=0.15):
    """
    Find the center of a circle in the image using Hough transform.

    Parameters
    ----------
    img : ndarray
        Input image.
    r : int
        Radius of the circle.
    theta_step : float, optional
        Step size for generating sine and cosine lookup tables.
        Smaller values result in more accurate but slower computation.
    threshold_factor : float, optional
        Factor multiplied by the maximum gradient magnitude to set the threshold.
        Adjusting this value can affect the sensitivity of circle detection.

    Returns
    -------
    tuple
        Coordinates (x, y) of the circle center.
    """

    # Gaussian blur to smooth the image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Get the dimensions of the image
    height, width = img.shape

    # Create sine and cosine lookup tables for circle center calculation
    sin_table = np.sin(np.arange(0, 2 * np.pi, theta_step))
    cos_table = np.cos(np.arange(0, 2 * np.pi, theta_step))

    # Compute the gradients of the image
    dx, dy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    dx2, dy2 = np.square(dx), np.square(dy)

    # Set the threshold to determine potential circle centers
    threshold = threshold_factor * np.max(np.sqrt(dx2 + dy2))

    # Create a binary mask where gradients below the threshold are set to 0 and others to 1
    mask = np.where(np.sqrt(dx2 + dy2) > threshold, 1, 0)

    # Find all non-zero pixels in the mask
    rows, cols = np.where(mask)

    # Create a Hough space for circle center voting
    hough_space = np.zeros_like(img, dtype=np.float64)

    # Traverse all possible circle center positions
    for i, theta_idx in enumerate(range(len(sin_table))):
        # Compute the circle radius
        a = rows - r * sin_table[theta_idx]
        b = cols + r * cos_table[theta_idx]
        ai = a.astype(int)
        bi = b.astype(int)

        # Keep only the points within the image range
        valid = (ai >= 0) & (bi >= 0) & (ai < height) & (bi < width)

        # Compute the voting value for each point to the circle center
        dist = np.sqrt(np.square(rows[valid] - ai[valid]) + np.square(cols[valid] - bi[valid]))
        vote = np.exp(-np.square(dist / (r * 0.1)))

        # Accumulate the voting results for this circle center position in the Hough space
        hough_space[ai[valid], bi[valid]] += vote

    # Get the maximum voting value
    max_vote = np.max(hough_space)

    # Find the circle center with the maximum votes
    x, y = np.unravel_index(np.argmax(hough_space), hough_space.shape)

    # If multiple potential circle centers are too close, take their average as the center
    distances = np.sqrt(np.square(rows - x) + np.square(cols - y))
    avg_distance = np.mean([distance for distance in distances if distance < 2 * r])
    indices = np.where(distances <= 2 * avg_distance)[0]

    x, y = np.average(rows[indices]), np.average(cols[indices])

    return x, y

def hough_circle_transform(data, radius=None, flag=False, theta_step=0.01, threshold_factor=0.15):
    """
    Detect circles in the input data using the Hough Circle Transform.

    This method filters the input data, computes gradients, and then applies the Hough Circle Transform to detect circles.
    If a circle is detected, it returns the coordinates of the circle center and its radius. If no circle is detected,
    it returns 0 for both the coordinates and the radius.

    Parameters
    ----------
    data : ndarray
        The input image data on which circle detection is to be performed.
    radius : float, optional
        The radius of the circle. If not provided, it will be estimated.
    flag : bool, optional
        A flag indicating whether to apply an additional filtering step to the input data. Defaults to False.
    theta_step : float, optional
        The step size used to generate the sine and cosine lookup tables in the Hough Circle Transform.
        Smaller values result in more precise but slower computations. Defaults to 0.01.
    threshold_factor : float, optional
        A factor that multiplies the maximum gradient magnitude to set the threshold for circle detection.
        Adjusting this value can affect the sensitivity of circle detection. Defaults to 0.15.

    Returns
    -------
    tuple
        A tuple (x, y, radius) where:
        x : float
            The x-coordinate of the detected circle center. Returns 0 if no circle is detected.
        y : float
            The y-coordinate of the detected circle center. Returns 0 if no circle is detected.
        radius : float
            The radius of the detected circle. Returns 0 if no circle is detected.

    Note:
        This function assumes the existence of additional functions like 'filtering', 'getAngle', 'getX',
        'least_squares_circle', and 'find_circle_center' which are not implemented here.
        These functions are responsible for preprocessing, gradient calculation, circle candidate generation,
        and final circle detection respectively.
    """
    if radius is None:
        # Apply filtering
        binary = filtering(data)

        # Compute gradient and angle
        grandien_img, angle = getAngle(binary)

        # Get potential circle center candidates
        x_center = getX(grandien_img)
        a = np.array(x_center)
        # If no circle center candidates are found, return zeros
        if len(a) == 0:
            x = 0
            y = 0
            radius = 0
        else:
            # Least squares fitting to estimate initial circle parameters
            xc, yc, radius, residual = least_squares_circle(a)

            # If the radius is zero, return zeros
            if radius == 0:
                x = 0
                y = 0
                radius = 0
            else:
                if flag == True:
                    # Filtering
                    data = filtering(data)
                    # Refine circle center using Hough transform
                y, x = find_circle_center(data, radius, theta_step=theta_step, threshold_factor=threshold_factor)
    else:
        if flag == True:
            # Filtering
            data = filtering(data)
            # Refine circle center using Hough transform
        y, x = find_circle_center(data, radius, theta_step=theta_step, threshold_factor=threshold_factor)


    return x, y, radius

def sx_zh(matrix):
    """
    Transpose the given matrix.

    Parameters
    ----------
    matrix : list of lists
        The input matrix.

    Returns
    -------
    list of lists
        The transposed matrix.

    """
    length = len(matrix)
    for j in range(length):
        for i in range(length // 2):
            # Swap positions based on coordinates
            matrix[j][i], matrix[j][length - i - 1] = matrix[j][length - i - 1], matrix[j][i]
    return matrix

def remove_low_frequency_noise(x, y, threshold_factor=0.1, enhancement_factor=0.5):
    """
    Remove low-frequency noise from the input signal.

    Parameters
    ----------
    x : array_like
        Input array representing x coordinates.
    y : array_like
        Input array representing y coordinates.
    threshold_factor : float, optional
        Factor to adjust the threshold for noise removal. The default is 0.1.
    enhancement_factor : float, optional
        Factor to control the amount of enhancement applied to the signal. The default is 0.5.

    Returns
    -------
    x_filtered : array_like
        Filtered x coordinates.
    y_filtered : array_like
        Filtered y coordinates.
    """

    # Calculate the magnitude array m
    m = np.sqrt(x ** 2 + y ** 2)
    # Check for and handle NaN values
    m[np.isnan(m)] = 0
    # Reshape m into a 2D array
    m_2d = m.reshape(-1, 1)

    # Convert m_2d to frequency domain representation using Fourier transform
    m_freq = np.fft.fftshift(np.fft.fft2(m_2d))

    # Calculate the dynamic threshold
    threshold = threshold_otsu(abs(m_freq)) * threshold_factor

    # Set low-frequency components to zero
    m_freq_filtered = m_freq.copy()
    m_freq_filtered[abs(m_freq) < threshold] = 0

    # Perform frequency domain enhancement
    m_freq_enhanced = m_freq * enhancement_factor + m_freq_filtered * (1 - enhancement_factor)

    # Convert back to spatial domain using inverse Fourier transform
    m_2d_filtered = np.fft.ifft2(np.fft.ifftshift(m_freq_enhanced)).real

    # Reshape m_2d_filtered back into a 1D array
    m_filtered = m_2d_filtered.flatten()

    # Calculate phase information
    phase = np.arctan2(y, x)

    # Recalculate x and y values based on the filtered m_filtered and phase information
    x_filtered = m_filtered * np.cos(phase)
    y_filtered = m_filtered * np.sin(phase)

    return x_filtered, y_filtered

def radial_gradient_maximum_optimized(img, blob, radius, num_rings=60, ring_range=2, num_points=100, threshold=1):

    """
    Find the maximum radial gradient points around a specified center in the image.

    Parameters:
    img (numpy.ndarray): The input image.
    blob (list): A single coordinate in the form [x, y] representing the center to be detected.
    radius (float): The radius of the circle.
    num_rings (int, optional): The number of rings. Defaults to 60.
    ring_range (int, optional): The range of the rings. Defaults to 2, indicating ±2 pixels around the original center coordinates to search for the maximum radial gradient points.
    num_points (int, optional): The number of sampling points on each ring. Defaults to 100.
    threshold (float, optional): The threshold for outliers to filter out. Defaults to 1.

    Returns:
    list: The adjusted center coordinates in the form [x, y].
    """
    # Get the height and width of the img image
    h, w = img.shape
    # Adjust the radius by multiplying it by a factor
    adj_radius = radius * 1
    # Generate an array of ring radii within the adjusted radius range
    ring_radius = np.linspace(adj_radius * 0.8, adj_radius * 1.2, num_rings)
    # Generate an array of angles for sampling points between 0 and 2π
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # Store the pixel value differences for each sampling point around the center
    ind_list = []

    # Search for the maximum radial gradient points within the specified range
    for ca in range(-ring_range, ring_range + 1):
        for cb in range(-ring_range, ring_range + 1):
            # Calculate the current center coordinates
            cur_row, cur_col = blob[1] + ca, blob[0] + cb
            # Store the sum of pixel values for each sampling point around the current center
            cacb_rn = np.empty(num_rings)

            # Iterate through each ring
            for i in range(num_rings):
                # Calculate the row and column coordinates of the sampling points on the ring
                row_coor = np.array([cur_row + ring_radius[i] * np.sin(theta) + 0.5]).astype(int)
                col_coor = np.array([cur_col + ring_radius[i] * np.cos(theta) + 0.5]).astype(int)

                # Restrict the coordinates of the sampling points within the image range
                row_coor[row_coor >= h] = h - 1
                row_coor[row_coor < 0] = 0
                col_coor[col_coor >= w] = w - 1
                col_coor[col_coor < 0] = 0

                # Calculate the sum of pixel values for the sampling points
                int_sum = np.sum(img[row_coor, col_coor])
                cacb_rn[i] = int_sum

            # Multiply the pixel values for each ring by weights and calculate the difference
            cacb_rn[:num_rings // 2] *= np.linspace(1, num_rings // 2, num_rings // 2)
            cacb_diff = np.sum(cacb_rn[:num_rings // 2]) - np.sum(cacb_rn[num_rings // 2:])
            ind_list.append([cur_col, cur_row, cacb_diff])

    # Convert the result to a numpy array
    ind_list = np.array(ind_list)
    # Find the center with the maximum difference
    ind_max = np.argmax(ind_list[:, 2])
    # Extract the center coordinates corresponding to the maximum difference
    adjusted_center = ind_list[ind_max]

    # Check for outliers
    z = np.abs(stats.zscore(ind_list[:, 2]))
    outlier = np.where(z > threshold)
    if len(outlier[0]) > 0:
        for each in outlier[0]:
            # If the distance between the center corresponding to the outlier and the image center is greater than the radius r, delete it
            if np.linalg.norm(ind_list[each, :2] - [h // 2, w // 2]) > radius:
                ind_list = np.delete(ind_list, outlier[0], axis=0)
                adjusted_center = ind_list[np.argmax(ind_list[:, 2])]

    return adjusted_center[:2]
    
  
def radial_gradient_maximum_optimized1(sample, blob, radius, num_rings=60, ring_range=2, num_points=80, threshold=1,
                                      step_size=0.00001):
    h, w = sample.shape
    adj_radius = radius * 1
    ring_radius = np.linspace(adj_radius * 0.8, adj_radius * 1.2, num_rings)
    theta = np.linspace(0, 2 * np.pi, num_points)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    def search_around_blob(blob, search_range, step):
        ca_range = np.arange(-search_range, search_range + step, step)
        cb_range = np.arange(-search_range, search_range + step, step)
        ind_list = []

        for ca in ca_range:
            for cb in cb_range:
                cur_row, cur_col = blob[1] + ca, blob[0] + cb
                row_coor = np.clip((cur_row + ring_radius[:, None] * sin_theta + 0.5).astype(int), 0, h - 1)
                col_coor = np.clip((cur_col + ring_radius[:, None] * cos_theta + 0.5).astype(int), 0, w - 1)

                int_sum = np.sum(sample[row_coor, col_coor], axis=1)
                weighted_sum = np.sum(int_sum[:num_rings // 2] * np.linspace(1, num_rings // 2, num_rings // 2))
                cacb_diff = weighted_sum - np.sum(int_sum[num_rings // 2:])

                ind_list.append([cur_col, cur_row, cacb_diff])

        return np.array(ind_list)

    def multi_level_search(blob, initial_step):
        best_point = blob
        search_range = ring_range
        step = initial_step

        ind_list = search_around_blob(best_point, search_range, step)
        ind_max = np.argmax(ind_list[:, 2])
        best_point = ind_list[ind_max, :2]

        while step > step_size:
            search_range = step
            step /= 10
            ind_list = search_around_blob(best_point, search_range, step)
            ind_max = np.argmax(ind_list[:, 2])
            best_point = ind_list[ind_max, :2]

        return np.append(best_point, ind_list[ind_max, 2])

    best_point = multi_level_search(blob, 0.1)
    adjusted_center = best_point

    return adjusted_center

def Jxtd1(data, flag=False, blobs=None, radius=None, num_rings=40, ring_range=2, num_points=100, threshold=1,step_size=0.00001):

    if blobs is None or radius is None:
        binary = filtering(data)
        grandien_img, angle = getAngle(binary)
        x_center = getX(grandien_img)
        a = np.array(x_center)
        if len(a) == 0:
            return 0, 0, 0
        else:
            #yc, xc, radius = least_squares_circle(a)

            # ret, binary1 = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #yc, xc, radius = find_circle_in_image(binary)
            data = (((data - data.min()) / (data.max() - data.min())) * 255).astype(np.uint8)
            xc, yc, radius = edge_detection_fitting_circles(data,flag=True)
            
            blobs = [xc, yc]
            #print(blobs)
            if blobs is None or radius is None:
                return 0, 0, 0
    if flag:
        # Filtering
        data = filtering(data)
    #print(radius)
    #data = np.log1p(data / data.max() * 10000)
    ref_ctr = radial_gradient_maximum_optimized1(data, blobs, radius, num_rings=num_rings, ring_range=ring_range, num_points=num_points, threshold=threshold,step_size=step_size)
    ref_blobs_list = ref_ctr

    return ref_blobs_list[0], ref_blobs_list[1], radius
 
def Jxtd(data,flag=False, blob=None, radius=None,num_rings=40, ring_range=2, num_points=100, threshold=2):

    """  
    Performs a series of image processing steps to detect and refine the center and radius of a circular object in an image.  
  
    Parameters:  
    - data: The input image data, expected to be a 2D array (e.g., grayscale image).  
    - flag (bool, optional): If True, forces the image to be filtered before further processing. Defaults to False.  
    - blob (list): A single coordinate in the form [x, y] representing the center to be detected.  
    - radius (float, optional): Initial radius estimate. If None, it will be estimated based on blob detection.  
    - num_rings (int, optional): Number of rings used for radial gradient analysis. Higher values provide finer gradient analysis.  
    - ring_range (int, optional): Range of pixels to sample around each ring for gradient calculation. Larger values may increase noise sensitivity.  
    - num_points (int, optional): Number of points sampled per ring for gradient analysis. More points improve accuracy but increase computation.  
    - threshold (float, optional): Threshold for detecting local maxima in the radial gradient. Higher values make detection stricter.  
  
    Function Steps and Impacts:  
    1. If blob or radius are not provided, automatically detect the blob center and radius using gradient and least squares circle fitting.  
    2. Optionally, if flag is True, apply filtering to the input image to enhance or preprocess it for better analysis.  
    3. Perform radial gradient maximum analysis on the image, using the provided or detected blob and radius as starting points.  
       - This step refines the blob center location by searching for local maxima in the radial gradient.  
    4. Return the refined blob center coordinates (y, x) and the estimated radius.  
  
    Returns:  
    - tuple: A tuple containing the refined y-coordinate of the blob center, the refined x-coordinate of the blob center, and the estimated radius.  
  
    Note: The return order of coordinates is (y, x) which is a common convention in image processing, where the origin (0,0) is at the top-left corner.  
    """  
    #If blob or radius are not provided, automatically detect them  
    if blob is None or radius is None:
        # Filtering
        binary = filtering(data)
        # Calculate gradient magnitude and angle
        gradient_img, angle = getAngle(binary)
        # Get x coordinates of edge points
        x_center = getX(gradient_img)
        a = np.array(x_center)
        if len(a) == 0:
            return 0,0,0
        else:
                  
            # Fit a circle to the edge points
            #yc, xc, radius, residual = least_squares_circle(a)
            #yc, xc, radius = find_circle_in_image(binary)
            data = (((data - data.min()) / (data.max() - data.min())) * 255).astype(np.uint8)
            xc, yc, radius = edge_detection_fitting_circles(data,flag=True)
            blob=[xc,yc]
            radius=radius
            if blob is None or radius is None:
                return 0, 0, 0
    if flag == True:
        # Filtering
        data = filtering(data)
    # Find the adjusted center using radial gradient maximum
    ref_ctr = radial_gradient_maximum_optimized(data, blob, radius, num_rings=num_rings, ring_range=ring_range, num_points=num_points, threshold=threshold)

    
    return ref_ctr[0], ref_ctr[1], radius
    
def plot():
    """
    Configure the plot settings.
    """
    # Set the figure size and style
    plt.style.use('_mpl-gallery-nogrid')

    # Modify the font and size of the axes
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)

    # Set square axes and display minus sign
    plt.axis('square')
    plt.rcParams['axes.unicode_minus'] = False

    # Resolve overlapping titles when plotting
    plt.tight_layout()
    #plt.figure(dpi=1200)
    # Display the plot
    plt.show()

def calc_radius(xc, yc, x, y):
    """
    Calculate the distance between a point (x, y) and the circle center (xc, yc).
    
    Parameters:
    xc, yc : float
        The x and y coordinates of the circle center.
    x, y : float or array-like
        The x and y coordinates of the point(s).
    
    Returns:
    float or ndarray
        The distance from the point(s) to the center.
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

def residuals(params, data):
    """
    Calculate the residuals between the actual distances and the radius of the circle.

    Parameters:
    params : list or array-like
        The parameters of the circle: [xc, yc, r], where (xc, yc) is the center and r is the radius.
    data : ndarray
        An array of shape (n, 2) where each row is a point (x, y) on the circle.

    Returns:
    ndarray
        The difference between the calculated radii from the center to each point and the given radius.
    """
    xc, yc, r = params
    x, y = data[:, 0], data[:, 1]
    return calc_radius(xc, yc, x, y) - r

def fit_circle(data):
    """
    Fit a circle to a set of points using least squares minimization.

    Parameters:
    data : ndarray
        An array of shape (n, 2) containing the (x, y) coordinates of the points.

    Returns:
    tuple
        A tuple containing the coordinates of the circle center (xc, yc) and the radius r.
        If fitting fails, returns (0, 0, 0).
    """
    x = data[:, 0]
    y = data[:, 1]

    # Initial guess for the circle's center and radius
    x_m = np.mean(x)
    y_m = np.mean(y)
    r_guess = np.mean(calc_radius(x_m, y_m, x, y))
    initial_guess = [x_m, y_m, r_guess]

    # Objective function to minimize (sum of squared residuals)
    def objective(params):
        return np.sum(residuals(params, data) ** 2)

    # Minimize the objective function
    result = minimize(objective, initial_guess, method='L-BFGS-B')

    if not result.success:
        return 0, 0, 0

    xc, yc, r = result.x
    return xc, yc, r

def detect_edges(binary_image):
    """
    Detect edges in a binary image using the Canny edge detector.

    Parameters:
    binary_image : ndarray
        A binary image (2D array) where edges are to be detected.

    Returns:
    ndarray
        A binary image where the edges are marked.
    """
    # Calculate the median of the image to determine thresholding bounds
    v = np.median(binary_image)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(binary_image, lower, upper)
    return edges

def extract_edge_points(edges):
    """
    Extract coordinates of edge points from the edge-detected image.

    Parameters:
    edges : ndarray
        A binary image containing edges detected by Canny or other edge detection methods.

    Returns:
    ndarray
        An array of shape (n, 2) containing the (x, y) coordinates of the edge points.
    """
    # Find non-zero points in the edge image, corresponding to edge locations
    points = np.column_stack(np.nonzero(edges))
    return points

def find_circle_in_image(binary):

    """
    Find a circle in a binary image by detecting edges and fitting a circle to the edge points.

    Parameters:
    binary : ndarray
        A binary image containing the object of interest.

    Returns:
    tuple
        A tuple containing the coordinates of the circle center (xc, yc) and the radius r.
        If no valid circle is found, returns (0, 0, 0).
    """
    # Detect edges in the binary image
    edges = detect_edges(binary)

    # Extract coordinates of edge points
    edge_points = extract_edge_points(edges)

    if len(edge_points) < 5:  # Too few points to fit a circle
        return 0, 0, 0

    # Fit a circle to the edge points
    xc, yc, r = fit_circle(edge_points)
    return xc, yc, r
    
def extract_region(A, B, C):

    """
    Function to extract a rectangular region from a 2D array (matrix) based on specified parameters.

    Parameters:
    A (tuple): A tuple of two integers (x, y) representing the starting point (upper-left corner) of the region to be extracted.
    B (int or list):
      - If an integer, it represents the side length of the square region to be extracted from the starting point A.
      - If a list of two integers, it represents the coordinates (x2, y2) of the opposite corner (lower-right corner) of the rectangular region to be extracted.
    C (numpy.ndarray): A 2D numpy array (matrix) from which the region is to be extracted.

    Returns:
    numpy.ndarray: A 2D numpy array containing the extracted region from the input array C. The shape of the returned array depends on the parameters A and B.

    Raises:
    ValueError: If parameter B is not an integer or a list of two integers, or if the coordinates in A and B are equal on the X or Y axis when B is a list.
    IndexError: If the specified region exceeds the bounds of the input array C.

    """
    x, y = A

    if isinstance(B, int):
        side_length = B
        left = x
        right = x + side_length
        top = y
        bottom = y + side_length
    elif isinstance(B, list) and len(B) == 2:
        x2, y2 = B
        if x == x2 or y == y2:
            raise ValueError(
                "The coordinates in A and B cannot be equal on the X or Y axis when B represents a corner.")
        left = min(x, x2)
        right = max(x, x2)
        top = min(y, y2)
        bottom = max(y, y2)
    else:
        raise ValueError(
            "Parameter B must be an integer (representing side length) or a list of two integers (representing coordinates).")

        # Check if the boundaries exceed the array dimensions
    if left < 0 or right > C.shape[0] or top < 0 or bottom > C.shape[1]:
        raise IndexError("The specified region exceeds the bounds of the array.")

        # Extract the region
    region = C[left:right, top:bottom]

    return region
    
def process_4d_array(A, B, array_4d):

    """
    Function to process a 4D array by extracting regions from each 2D slice (3rd and 4th dimensions) based on given parameters.  

    Parameters:  
    A (tuple): A tuple of two integers (x, y) representing the starting point (upper-left corner) of the region to be extracted from each 2D slice.  
    B (int or tuple):   
      - If an integer, it represents the side length of the square region to be extracted from the starting point A.  
      - If a tuple of two integers, it represents the coordinates (x2, y2) of the opposite corner (lower-right corner) of the rectangular region to be extracted.  
    array_4d (numpy.ndarray): A 4D numpy array with shape (d1, d2, height, width) where d1 and d2 are the first two dimensions, and height and width are the dimensions of the 2D slices.  

    Returns:  
    numpy.ndarray: A 4D numpy array with shape (d1, d2, region_height, region_width) where region_height and region_width are determined by the shape of the extracted regions.  
      This array contains the extracted regions from each 2D slice of the input array_4d. If an error occurs during extraction, the corresponding slice in the output array will be filled with zeros.  

    Raises:  
    No exceptions are explicitly raised by this function, but it handles IndexError and ValueError by printing error messages and filling the corresponding slice with zeros.  
    """
    
    d1, d2, _, _ = array_4d.shape  # Extract the first two dimensions of the 4D array  

    # Determine the shape of the region to be extracted  
    if isinstance(B, int):
        region_shape = (B, B)  # Square region  
    else:
        region_shape = (
        abs(B[0] - A[0]), abs(B[1] - A[1]))  # Rectangular region based on the difference between A and B coordinates  

    # Initialize an array to store the extracted regions  
    extracted_regions = np.zeros((d1, d2, region_shape[0], region_shape[1]))

    # Iterate over each 2D slice in the 4D array  
    for i in range(d1):
        for j in range(d2):
            try:
                # Extract the region from the current 2D slice  
                region = extract_region(A, B, array_4d[i, j])
                # Store the extracted region in the output array  
                extracted_regions[i, j] = region
            except IndexError as e:
                # Print an error message if the region exceeds the slice boundaries  
                print(f"IndexError at position ({i}, {j}): {e}")
            except ValueError as e:
                # Print an error message if there's a problem with the parameters A or B  
                print(f"ValueError at position ({i}, {j}): {e}")

                # Return the array containing the extracted regions  
    return extracted_regions
    
def cut(z):
    # Get the number of rows (nl) and columns (nc) in the input array z
    nl, nc = z.shape

    # Return the array z without the first and last rows and columns
    return z[1:nl-1, 1:nc-1]
    
def plot_jxtd(sample, blob, radius, num_rings=10, ring_range=2, num_points=100, threshold=1, step_size=0.1):

    h, w = sample.shape
    adj_radius = radius * 1
    ring_radius = np.linspace(adj_radius * 0.8, adj_radius * 1.2, num_rings)
    theta = np.linspace(0, 2 * np.pi, num_points)
    adjusted_centers = []

    # Creating a subgraph
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Start Image
    ax = axes[0]
    ax.imshow(sample, cmap='gray')
    ax.set_title('Start')


    ind_list = []
    grid_values = []

    # Grid Search
    ax = axes[1]
    ax.set_xlim(blob[0] - ring_range, blob[0] + ring_range)
    ax.set_ylim(blob[1] - ring_range, blob[1] + ring_range)
    ax.invert_yaxis()
    ax.set_title('Grid Search')
    for ca in np.arange(-ring_range, ring_range + step_size, step_size):
        for cb in np.arange(-ring_range, ring_range + step_size, step_size):
            cur_row, cur_col = blob[1] + ca, blob[0] + cb
            ax.plot(cur_col, cur_row, 'r.', markersize=2)

    for ca in np.arange(-ring_range, ring_range + step_size, step_size):
        for cb in np.arange(-ring_range, ring_range + step_size, step_size):
            cur_row, cur_col = blob[1] + ca, blob[0] + cb
            cacb_rn = np.empty(num_rings)
            for i in range(num_rings):
                row_coor = np.array([cur_row + ring_radius[i] * np.sin(theta) + 0.5]).astype(int)
                col_coor = np.array([cur_col + ring_radius[i] * np.cos(theta) + 0.5]).astype(int)
                row_coor[row_coor >= h] = h - 1
                row_coor[row_coor < 0] = 0
                col_coor[col_coor >= w] = w - 1
                col_coor[col_coor < 0] = 0
                int_sum = np.sum(sample[row_coor, col_coor])
                cacb_rn[i] = int_sum

            # Intermediate process
            if len(ind_list) == 0:  # Record only once
                ax = axes[2]
                ax.imshow(sample, cmap='gray')
                ax.set_title('Intermediate')
                for j in range(num_rings):
                    ring_theta = np.linspace(0, 2 * np.pi, 100)
                    ring_x = blob[0] + ring_radius[j] * np.cos(ring_theta) # Draw only one of the points
                    ring_y = blob[1] + ring_radius[j] * np.sin(ring_theta)
                    color = plt.cm.hsv(j / num_rings)
                    ax.plot(ring_x, ring_y, color=color, linewidth=0.15)

            # Recording Grid Values
            grid_values.append((cur_row, cur_col, cacb_rn[:num_rings // 2].sum(), cacb_rn[num_rings // 2:].sum()))

            cacb_rn[:num_rings // 2] *= np.linspace(1, num_rings // 2, num_rings // 2)
            cacb_diff = np.sum(cacb_rn[:num_rings // 2]) - np.sum(cacb_rn[num_rings // 2:])
            ind_list.append([cur_row, cur_col, cacb_diff])

    ind_list = np.array(ind_list)
    ind_max = np.argmax(ind_list[:, 2])
    adjusted_centers.append(ind_list[ind_max])

    # Calculation result image
    fig.delaxes(axes[3])  # Delete the original subgraph
    ax = fig.add_subplot(1, 5, 4, projection='3d')
    ax.set_title('Integral Differences')
    x = ind_list[:, 1]
    y = ind_list[:, 0]
    z = ind_list[:, 2]
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.scatter(ind_list[ind_max][1], ind_list[ind_max][0], ind_list[ind_max][2], color='r', s=350, label='Max Diff')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Integral Difference')
    ax.legend()

    adjusted_centers = np.array(adjusted_centers)
    z = np.abs(stats.zscore(adjusted_centers[:, 2]))
    outlier = np.where(z > threshold)
    if len(outlier[0]) > 0:
        for each in outlier[0]:
            if np.linalg.norm(adjusted_centers[each, :2] - [h // 2, w // 2]) > radius:
                adjusted_centers = np.delete(adjusted_centers, outlier[0], axis=0)

    # Final Image
    ax = axes[4]
    ax.imshow(sample, cmap='gray')
    for center in adjusted_centers:
        ax.plot(center[1], center[0], 'go')
    ax.set_title('Final Adjusted Centers')

    plt.tight_layout()
    plt.show()

    return adjusted_centers[:, :2][0][1],adjusted_centers[:, :2][0][0]
    
def find_circle_center_and_plot(img, r, theta_step=0.01, threshold_factor=0.35):
    # Gaussian blur to smooth the image
    img_smoothed = cv2.GaussianBlur(img, (5, 5), 0)

    # Get the dimensions of the image
    height, width = img.shape

    # Create sine and cosine lookup tables for circle center calculation
    sin_table = np.sin(np.arange(0, 2 * np.pi, theta_step))
    cos_table = np.cos(np.arange(0, 2 * np.pi, theta_step))

    # Compute the gradients of the image
    dx, dy = cv2.Sobel(img_smoothed, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(img_smoothed, cv2.CV_64F, 0, 1, ksize=5)
    dx2, dy2 = np.square(dx), np.square(dy)

    # Set the threshold to determine potential circle centers
    gradient_magnitude = np.sqrt(dx2 + dy2)
    threshold = threshold_factor * np.max(gradient_magnitude)

    # Create a binary mask where gradients below the threshold are set to 0 and others to 1
    mask = np.where(gradient_magnitude > threshold, 1, 0)

    # Find all non-zero pixels in the mask
    rows, cols = np.where(mask)

    # Create a Hough space for circle center voting
    hough_space = np.zeros_like(img, dtype=np.float64)

    # Traverse all possible circle center positions
    for radius in np.arange(r - 1, r + 1 + 0.2, 0.2): 
        for i, theta_idx in enumerate(range(len(sin_table))):
            # Compute the circle radius
            a = rows - r * sin_table[theta_idx]
            b = cols + r * cos_table[theta_idx]
            ai = a.astype(int)
            bi = b.astype(int)

            # Keep only the points within the image range
            valid = (ai >= 0) & (bi >= 0) & (ai < height) & (bi < width)

            # Compute the voting value for each point to the circle center
            dist = np.sqrt(np.square(rows[valid] - ai[valid]) + np.square(cols[valid] - bi[valid]))
            vote = np.exp(-np.square(dist / r ))

            # Accumulate the voting results for this circle center position in the Hough space
            hough_space[ai[valid], bi[valid]] += vote

    # Get the maximum voting value
    max_vote = np.max(hough_space)

    # Find the circle center with the maximum votes
    x, y = np.unravel_index(np.argmax(hough_space), hough_space.shape)

    # If multiple potential circle centers are too close, take their average as the center
    distances = np.sqrt(np.square(rows - x) + np.square(cols - y))
    avg_distance = np.mean([distance for distance in distances if distance < 2*r])
    indices = np.where(distances <= 2*avg_distance)[0]

    x, y = np.average(rows[indices]), np.average(cols[indices])

    # Plot the results
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))

    # Original Image
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original Image', fontsize=10)

    # Smoothed Image
    axs[0, 1].imshow(img_smoothed, cmap='gray')
    axs[0, 1].set_title('Smoothed Image', fontsize=10)

    # Gradient Magnitude
    axs[0, 2].imshow(gradient_magnitude, cmap='gray')
    axs[0, 2].set_title('Gradient Magnitude', fontsize=10)

    # Mask
    axs[1, 0].imshow(mask, cmap='gray')
    axs[1, 0].set_title('Mask', fontsize=10)

    # Hough Space
    axs[1, 1].imshow(hough_space, cmap='gray')
    axs[1, 1].set_title('Hough Space', fontsize=10)

    # Scatter plot with histograms
    scatter_ax = axs[1, 2]
    scatter_ax.scatter(cols, rows, color='red', s=2)
    scatter_ax.set_title('Scatter plot of circle center points', fontsize=10)
    scatter_ax.invert_yaxis()

    # Histograms
    hist_x = axs[2, 0]
    hist_y = axs[2, 1]

    hist_x.bar(range(128), np.count_nonzero(hough_space, axis=0), color='blue', alpha=0.75)
    hist_x.set_title('Histogram of X coordinates', fontsize=10)
    hist_x.set_ylabel('Density')

    hist_y.barh(range(128), np.count_nonzero(hough_space, axis=1), color='blue', alpha=0.75)
    hist_y.set_title('Histogram of Y coordinates', fontsize=10)
    hist_y.set_xlabel('Density')

    # Detected Circle Center
    axs[2, 2].imshow(img, cmap='gray')
    axs[2, 2].scatter(y, x, color='red')
    axs[2, 2].set_title('Detected Circle Center', fontsize=10)

    plt.tight_layout()
    plt.show()

    return x, y,hough_space

def plot_hough_space_with_histograms(hough_space, region_size=32):
    # Find the brightest point in Hough space
    center_y, center_x = np.unravel_index(np.argmax(hough_space), hough_space.shape)

    # Define the region around the brightest point
    y_start = max(center_y - region_size // 2, 0)
    y_end = min(center_y + region_size // 2, hough_space.shape[0])
    x_start = max(center_x - region_size // 2, 0)
    x_end = min(center_x + region_size // 2, hough_space.shape[1])

    # Extract the region from the Hough space
    region_hough_space = hough_space[y_start:y_end, x_start:x_end]
    threshold = np.mean(region_hough_space)
    
    region_hough_space = np.where(region_hough_space > threshold, 1, 0)
    # Calculate the number of non-zero elements in each column and row within the region
    region_x_hist = np.count_nonzero(region_hough_space, axis=0)
    region_y_hist = np.count_nonzero(region_hough_space, axis=1)

    # Create histograms of length 128, filling with zeros where necessary
    x_hist = np.zeros(128, dtype=int)
    y_hist = np.zeros(128, dtype=int)

    # Determine the start positions to place the region's histograms into the full histograms
    x_hist_start = max(0, center_x - region_size // 2)
    y_hist_start = max(0, center_y - region_size // 2)

    # Insert the region's histogram data into the full histograms
    x_hist[x_hist_start:x_hist_start + len(region_x_hist)] = region_x_hist
    y_hist[y_hist_start:y_hist_start + len(region_y_hist)] = region_y_hist

    # Create graphs and subgraphs
    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(4, 4, figure=fig, wspace=0.01, hspace=0.01)

    # Draw the region from the Hough space matrix
    ax_main = fig.add_subplot(gs[1:, :-1])
    im = ax_main.imshow(hough_space, cmap='gray')

    # Add a title to the upper right corner of the sub-image and set the font color to white
    ax_main.text(0.95, 0.95, 'Hough Space', transform=ax_main.transAxes, fontsize=12, ha='right', color='white')

    # Draw a histogram along the x-axis
    ax_xhist = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_xhist.bar(range(128), x_hist, color='blue', width=0.6)
    ax_xhist.set_ylabel('Count')
    ax_xhist.get_xaxis().set_visible(False)

    # Customize the x-axis scale, excluding 0
    xticks = ax_xhist.get_yticks()
    ax_xhist.set_yticks(xticks[xticks != 0])

    # Draw a histogram along the y-axis
    ax_yhist = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_yhist.barh(range(128), y_hist, color='green', height=0.6)
    ax_yhist.set_xlabel('Count')
    ax_yhist.get_yaxis().set_visible(False)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)
    plt.show()

    
def plot_Corre(data1, data2, flag=False, step=5):
    pxh, pxw = data1.shape
    original_data1 = data1.copy()

    # If flag is True, data1 is filtered.
    if flag:
        data1 = circle.filtering(data1)

    # Calculate the cross correlation between data1 and data2
    correlation = signal.correlate2d(data1, data2, boundary="symm", mode='same')

    # Polynomial fitting is performed on the horizontal and vertical sums of the cross-correlation results.
    horizon = np.sum(correlation, axis=1)
    vertical = np.sum(correlation, axis=0)
    z1 = np.polyfit(np.arange(pxh), horizon, step)
    z2 = np.polyfit(np.arange(pxw), vertical, step)

    # Create a polynomial object using the fitted polynomial coefficients and find the first-order derivative
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)
    yyyd1 = np.polyder(p1, 1)
    yyyd2 = np.polyder(p2, 1)

    # Find the coordinates of the extreme points
    x_peaks = np.roots(yyyd1)
    y_peaks = np.roots(yyyd2)

    # Select the extreme point closest to the center of the image (here we assume that the center of the image is (pxh/2, pxw/2))
    x_peak = min(x_peaks.real, key=lambda x: abs(x - pxw / 2))
    y_peak = min(y_peaks.real, key=lambda x: abs(x - pxh / 2))

    # Draw images of each stage
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # original data1
    axs[0, 0].imshow(original_data1, cmap='gray')
    axs[0, 0].set_title('Original Data1')

    # Filtered data 1
    axs[0, 1].imshow(data1, cmap='gray')
    axs[0, 1].set_title('Filtered Data1' if flag else 'Data1')

    # 原original data2
    axs[0, 2].imshow(data2, cmap='gray')
    axs[0, 2].set_title('Data2')

    # Cross-correlation results
    axs[1, 0].imshow(correlation, cmap='gray')
    axs[1, 0].set_title('Cross-Correlation')

    # The sum of the horizontal and vertical
    axs[1, 1].plot(horizon, label='Horizon Sum')
    axs[1, 1].plot(p1(np.arange(pxh)), label='Polynomial Fit', linestyle='--')
    axs[1, 1].set_title('Horizon Sum and Polynomial Fit')
    axs[1, 1].legend()

    axs[1, 2].plot(vertical, label='Vertical Sum')
    axs[1, 2].plot(p2(np.arange(pxw)), label='Polynomial Fit', linestyle='--')
    axs[1, 2].set_title('Vertical Sum and Polynomial Fit')
    axs[1, 2].legend()

    plt.show()

    return y_peak, x_peak
    
