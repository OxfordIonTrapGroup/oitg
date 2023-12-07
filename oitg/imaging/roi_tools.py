import numpy as np
from scipy.ndimage import measurements
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math


def find_ion_centers(im_line, min_width=2, threshold=None, fit=False):
    """Find (approximate) location of the ions in a chain.
    im_line is a 1d array of fluorescence along the axis of the crystal (a "line
    image").
    To be detected each ion has to have values of more than "threshold" over at
    least min_width contiguous bins. By default, threshold is (max+min)/2

    If fit is False, returns a list of ion positions rounded to the nearest pixel
    If fit is True, fits a Gaussian to each position and returns the fitted (sub-pixel)
    center and the best fit function evaluation
    """
    if threshold is None:
        threshold = (np.amax(im_line) + np.amin(im_line)) / 2

    labels, num_features = measurements.label(im_line > threshold)

    x = np.arange(len(im_line))

    # Find the center position of each ion
    x_cens = []

    for i in range(num_features):
        xs = x[np.nonzero(labels == i + 1)]
        if max(xs) - min(xs) < min_width:
            continue
        x_cen = np.mean(xs)
        x_cens.append(x_cen)

    if not fit:
        # Return the centers quantised to the nearest pixel
        return x_cens

    N = len(x_cens)

    if N == 0:
        return [], [0] * len(im_line)

    # Fit a Gaussian to each peak
    def fit_func(x, *params):
        y = np.zeros(x.shape)
        for i in range(N):
            x0 = params[i]
            sigma = params[N + i]
            amp = params[2 * N + i]
            y += amp * np.exp(-(x - x0)**2 / sigma**2 / 2)
        return y

    popt, _ = curve_fit(fit_func,
                        x,
                        im_line,
                        p0=x_cens + [min_width] * N + [np.amax(im_line)] * N)

    x_cens_fit = popt[:N]
    y_fit = fit_func(x, *popt)
    return x_cens_fit, y_fit


def find_ion_regions(im, max_width=20):
    """Find bounded regions around ions in an image.
    The ion chain is assumed to lie along the first axis in the image.
    The image is assumed to be trimmed such that the axis perpendicular to the
    chain is tight.

    max_width: the maximum width of each ions region

    Returns a list of regions, each region is a tuple of
    the minimum and maximum extent of the ions signal.
    """

    im_line = np.sum(im, axis=1)
    im_line -= np.amin(im_line)

    threshold = (np.amax(im_line) + np.amin(im_line)) / 2

    x = np.arange(len(im_line))

    labels, num_features = measurements.label(im_line > threshold)

    # Find the centre position of each ion
    x_cens = [int(np.median(x[labels == i + 1])) for i in range(num_features)]

    # Find the lower and upper x position of each ion
    x_rois = []
    for i, x_c in enumerate(x_cens):
        # The leftmost part of the ROI is at bounded by max_width
        x_low = round(x_c - max_width / 2)
        if i > 0:
            # if this ion is not the leftmost, the ROI edge is half the distance to
            # the nearest ion or the max width
            x_low = max([x_low, math.ceil((x_cens[i - 1] + x_c) / 2)])
        x_high = round(x_c + max_width / 2)
        if i < len(x_cens) - 1:
            x_high = min([x_high, math.ceil((x_cens[i + 1] + x_c) / 2)])
        x_rois.append((x_low, x_high))

    return x_rois


def plot_ion_regions(im, x_rois, ax=None):
    """Plot the result of find_ion_regions"""
    if ax is None:
        ax = plt
    x = np.arange(im.shape[0])

    ax.imshow(im.transpose(), cmap="Greys")
    for roi in x_rois:
        ax.fill_between(x, [im.shape[1]] * len(x),
                        where=np.logical_and(x >= roi[0], x <= roi[1]),
                        alpha=0.2)


def trim_image(im, n_width=10, n_length=50):
    """Trims off background area around an image of an ion chain.
    n_length is the half-length of the selected area along axis 0, and n_width
    is the same for axis 1

    Return the sub-image, and the coordinate range tuples as
    [(x_min,x_max),(y_min,y_max)]"""
    coords = np.where(im == np.amax(im))
    x_cen, y_cen = [coords[i][0] for i in range(2)]

    x_range = x_cen - n_length, x_cen + n_length
    y_range = y_cen - n_width, y_cen + n_width

    im_sub = im[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    return im_sub, [x_range, y_range]
