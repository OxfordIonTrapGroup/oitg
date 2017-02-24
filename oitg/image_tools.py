
import numpy as np
import matplotlib.image as mpimg
import scipy.ndimage.filters
from oitg.fitting.gaussian_beam import gaussian_beam

# Size of a pixel in the image plane in micrometers
pixel_size = 5.2


def load_image(fileName, normalise=False):
    img = mpimg.imread(fileName)
    height, width = img.shape[0:2]

    # If the image is not mono, make it mono
    if len(img.shape) > 2:
        if img.shape[2] == 4:
            # The color data is most likely RGBA
            # Convert to mono by taking the average of RGA
            mono_img = np.mean(img[:, :, :-1].astype(float), axis=2)
        else:
            # not implemented
            print(img[0, 0, :])
            raise Exception('Conversion to mono not yet implemented '
                            'for this shape')
    else:
        mono_img = img.astype(float)

    if normalise:
        # Find the sum of all the pixels
        pixel_sum = np.sum(mono_img)

        # Normalise the image
        mono_img /= pixel_sum

    return (width, height), mono_img


def fit_slice(x, y, x0):

    p, p_error, x_fit, y_fit = \
        gaussian_beam.fit(x, y, evaluate_function=True,
                          initialise={'x0': x0})

    return p, p_error, x_fit, y_fit


def find_center(img):
    sigma = 10
    filtered = scipy.ndimage.filters.gaussian_filter(img, sigma)

    a, b = np.unravel_index(filtered.argmax(), filtered.shape)

    row_y = filtered[a, :]
    row_x = np.arange(len(row_y))
    row_p, row_p_error, row_x_fit, row_y_fit = fit_slice(row_x, row_y, b)

    col_y = filtered[:, b]
    col_x = np.arange(len(col_y))
    col_p, col_p_error, col_x_fit, col_y_fit = fit_slice(col_x, col_y, a)

    x0 = row_p['x0']
    wx = row_p['w0']

    y0 = col_p['x0']
    wy = col_p['w0']

    return x0, y0, wx, wy


def fit_image(img, make_outline=False):
    # Size of mask in units of w0
    mask_size = 1

    # Make an initial guess
    x0, y0, wx, wy = find_center(img)

    # Mask the data using the
    mask_lim_y = [y0 - mask_size*wy, x0 + mask_size*wy]

    mask_lim_x = np.array([x0 - mask_size*wx, x0 + mask_size*wx])
    mask_lim_x = np.around(mask_lim_x).astype(int)
    mask_lim_x = np.maximum(mask_lim_x, [0, 0])
    mask_lim_x = np.minimum(mask_lim_x, len(img[0, :]))
    mask_x = np.arange(mask_lim_x[0], mask_lim_x[1])

    mask_lim_y = np.array([y0 - mask_size*wy, y0 + mask_size*wy])
    mask_lim_y = np.around(mask_lim_y).astype(int)
    mask_lim_y = np.maximum(mask_lim_y, [0, 0])
    mask_lim_y = np.minimum(mask_lim_y, len(img[:, 0]))
    mask_y = np.arange(mask_lim_y[0], mask_lim_y[1])

    row_y = img[int(y0), mask_lim_x[0]:mask_lim_x[1]]
    row_x = mask_x

    row_p, row_p_error, row_x_fit, row_y_fit = fit_slice(row_x, row_y, x0)

    col_y = img[mask_lim_y[0]:mask_lim_y[1], int(x0)]
    col_x = mask_y

    col_p, col_p_error, col_x_fit, col_y_fit = fit_slice(col_x, col_y, y0)

    x0 = row_p['x0']
    wx = row_p['w0']
    ix = row_p['a']
    y0 = col_p['x0']
    wy = col_p['w0']
    iy = col_p['a']

    if make_outline:

        outlined_img = np.zeros(img.shape)
        lw = 20
        val = 255
        outlined_img[mask_lim_y[0]-lw/2:mask_lim_y[1]+lw/2,
                     mask_lim_x[0]-lw/2:mask_lim_x[0]+lw/2] = val
        outlined_img[mask_lim_y[0]-lw/2:mask_lim_y[1]+lw/2,
                     mask_lim_x[1]-lw/2:mask_lim_x[1]+lw/2] = val
        outlined_img[mask_lim_y[0]-lw/2:mask_lim_y[0]+lw/2,
                     mask_lim_x[0]-lw/2:mask_lim_x[1]+lw/2] = val
        outlined_img[mask_lim_y[1]-lw/2:mask_lim_y[1]+lw/2,
                     mask_lim_x[0]-lw/2:mask_lim_x[1]+lw/2] = val

        return img[mask_lim_y[0]:mask_lim_y[1],
                   mask_lim_x[0]:mask_lim_x[1]],\
            outlined_img,\
            row_x_fit, row_y_fit, row_x, row_y,\
            col_y_fit, col_x_fit, col_y, col_x,\
            wx*pixel_size, wy*pixel_size

    else:
        return img[mask_lim_y[0]:mask_lim_y[1], mask_lim_x[0]:mask_lim_x[1]], \
            row_x_fit, row_y_fit, row_x, row_y, \
            col_y_fit, col_x_fit, col_y, col_x, \
            wx*pixel_size, wy*pixel_size, \
            ix, iy


def sub_image(ax_min, ax_max, img, axis='x'):

    if axis == 'x':
        index = 1
    else:
        index = 0

    ax_size = img.shape[index]

    ax_min_pix = round(ax_min*ax_size)
    ax_max_pix = round(ax_max*ax_size)

    if axis == 'x':
        return img[:, ax_min_pix:ax_max_pix]
    else:
        return img[ax_min_pix:ax_max_pix, :]
