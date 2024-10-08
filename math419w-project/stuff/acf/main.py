import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.random import default_rng
from statsmodels.tsa import stattools


# import pandas as pd
# from statsmodels.tsa.ar_model import AutoReg
# from statsmodels.graphics import tsaplots
# from skimage import color, data, restoration
# from scipy.signal import convolve2d as conv2, convolve2d
# stuff I think is cool that I'm not using
# def denoise_weiner(img):
#     tmp = img.copy()
#     psf = np.ones((5, 5)) / 25
#     tmp = convolve2d(tmp, psf, 'same')
#     tmp += 0.1 * tmp.std() * np.random.standard_normal(tmp.shape)
#     deconvolved_img = restoration.unsupervised_wiener(tmp, psf)
#     return deconvolved_img

# first attempt at plotting using pandas, also generates a 'control' group from a given

# def make_plots(roi_coords, img, roi_name, direction, lag_limit=100, ac_limit=0.5, control=False, uncor_img=None):
#     """
#     Uses roi_coords to crop an img, then generates the autocorrelation for that using pandas.
#     :param roi_coords: coordinates in (x,y,w,h) that give the roi we are cropping
#     :param img: numpy, opencv type image
#     :param roi_name: name for the title
#     :param direction: 'x' or 'y'
#     :param lag_limit: number of lags
#     :param ac_limit: y-axis limit
#     :param control: (optional) do uncorrelated and correlated control groups, given an image the size of img,
#                     presumably filled with noise, using the roi_coords. Uses AR(10) for correlation
#     :param uncor_img: (optional) the uncorrelated image
#     :return: roi_crop, the cropped img.
#     """
#     roi_crop = create_cropped_img(img, roi_coords)
#     if control:
#         noise_sections = make_sections(roi_coords, uncor_img, direction)
#     roi_sections = make_sections(roi_coords, img, direction)
#     for section in roi_sections:
#         roi_plot = pd.plotting.autocorrelation_plot(section.flatten())
#     roi_plot.set_ybound(-ac_limit, ac_limit)
#     roi_plot.set_xlim(0, lag_limit)
#     # roi_plot.autoscale()
#     roi_plot.grid()
#     roi_plot.set_title(roi_name + "sections")
#     plt.savefig(roi_name + ".png")
#     plt.show()
#     if control:
#         for section in noise_sections:
#             noise_plot = pd.plotting.autocorrelation_plot(section.flatten())
#         noise_plot.set_ybound(-ac_limit, ac_limit)
#         noise_plot.set_xlim(0, lag_limit)
#         noise_plot.set_title(roi_name + "uncorrelated sections")
#         plt.show()
#         for section in noise_sections:
#             model = AutoReg(section.flatten(), lags=10).fit()
#             ar10 = model.forecast(50)
#             corr_plot = pd.plotting.autocorrelation_plot(ar10)
#         corr_plot.set_ybound(-ac_limit, ac_limit)
#         corr_plot.set_xlim(0, lag_limit)
#         corr_plot.set_title(roi_name + "correlated sections")
#         plt.show()
#
#     return roi_crop


def annotate_image(coords, img, color):
    """
    used to bounding box annotate image with given coordinates, usually ROI.
    :param coords: (x,y,w,h)
    :param img: image to draw on
    :param color: color to use defined as (R,G,B) from 0-255, eg: red is (255,0,0)
    :return: a joke
    """
    x, y, w, h = coords
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return None


def avg_sections(array):
    """
    given a column by row array, average each row.
    :param array:
    :return: avg_rows
    """
    avg_rows = np.mean(array, axis=0)
    return avg_rows


def create_acf_sections(sections):
    """
    from a list of sections, performs autocorrelation, with confidence interval (lower and upper), and ratios between
    successive lags, for each section.
    :param sections: list of sections created from create_sections()
    :return: acf_sects, r_sects, conf_l_sect, conf_u_sect
    """
    acf_sects, conf_l_sect, conf_u_sect, r_sects = [], [], [], []
    for section in sections:
        acf, confint = (stattools.acf(section.flatten(), alpha=0.05))
        acf_sects.append(acf)
        r = ratios(acf[0:5])
        r_sects.append(r)
        confint_0, confint_1 = np.split(confint.transpose(), confint.transpose().shape[0])
        conf_l_sect.append(confint_0.flatten())
        conf_u_sect.append(confint_1.flatten())
    acf_sects = np.array(acf_sects)
    conf_l_sect = np.array(conf_l_sect)
    conf_u_sect = np.array(conf_u_sect)
    r_sects = np.array(r_sects)
    return acf_sects, r_sects, conf_l_sect, conf_u_sect


def create_acf_plot(acf, ratio, conf_l, conf_u, title):
    """
    plots acf given acf, ratio, confidence interval, and a title
    :param acf:
    :param ratio:
    :param conf_l: confidence lower bound
    :param conf_u: confidence upper bound
    :param title:
    :return: fig
    """
    lags = list(range(len(acf)))
    exp_x = np.arange(0, 100, 0.1)
    exp_y = ratio ** exp_x
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stem(lags, acf)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_xlim(-1, len(acf))
    ax.plot(exp_x, exp_y)
    lags = lags[1:]
    conf_l = conf_l[1:]
    conf_u = conf_u[1:]
    acf = acf[1:]
    lags[0] -= 0.5
    lags[-1] += 0.5
    ax.fill_between(lags, conf_l - acf, conf_u - acf, alpha=0.25, linewidth=0)
    plt.title(title)
    plt.show()
    return fig


def create_avg_acf_plot(roi_coords, title, img, direction):
    """
    At this point I got lazy, just takes all the previously nicely seperated functions and throws them in here + plot.
    Given roi_coords, img, direction: Create a cropped img. Then create subsections of that cropped image in x or y
    direction. Then calculate autocorrelation of each of those sections. Then averages each of those acf with each
    other. Then plots that + an exponential regression + confidence intervals.
    The plotting is a half-baked implementation of the "_plot_corr" from statsmodel.tsaplots.
    :param roi_coords: region of interest coordinates
    :param title: title of plot
    :param img: image that region of interest is in
    :param direction: x or y sections
    :return: acf_plot
    """
    roi_img = create_cropped_img(roi_coords, img)
    roi_sections = create_sections(roi_img, direction)
    acf_sect, r_sect, c_sect_l, c_sect_u = create_acf_sections(roi_sections)
    acf_avg, c_sect_l_avg, c_sect_u_avg = avg_sections(acf_sect), avg_sections(c_sect_l), avg_sections(c_sect_u)
    r_avg = r_sect.mean()
    acf_plot = create_acf_plot(acf_avg, r_avg, c_sect_l_avg, c_sect_u_avg, title)
    acf_plot.savefig(title.replace(" ", "_") + ".png")
    return acf_plot


def create_cropped_img(coords, img):
    """
    create a cropped image with given coordinates
    :rtype: numpy.array()
    :param coords: (x,y,w,h)
    :param img: image to create cropped copy from
    :return: cropped_img
    """
    x, y, w, h = coords
    cropped_img = img[y:y + h, x: x + w]
    return cropped_img


def create_sections(cropped_img, direction):
    if direction == "x":
        roi_sections = np.split(cropped_img, cropped_img.shape[0])
    if direction == "y":
        roi_sections = np.hsplit(cropped_img, cropped_img.shape[1])
    return roi_sections


def ratios(acf_value):
    """
    successive ratios between lags for generating an exponential function.
    :param acf_value: values from acf presumably, although any data would work.
    :return: ratio
    """
    ratio = acf_value[1:] / acf_value[:-1]
    return ratio


plt.interactive(True)
img_orig = cv2.imread('B4-058.tif', 0)

# noise reduction filter, because I don't want to look at the noise.
img_blurred = cv2.medianBlur(img_orig, 3)
img_diff = img_orig + 0.0 - img_blurred

# generating a noisy image
mu = np.mean(img_diff)
sigma = np.std(img_diff)
uncor_img = np.random.normal(0, sigma, img_orig.shape)
# could also make binary

# number of cross-sections:
p_sect = 10

# these are the horizontal cuts.
x_cross_coords = (0, 1000, img_diff.shape[1], p_sect)
x_background_coords = (0, 2000, 700, p_sect)
x_nucleus_coords = (467, 929, 400, p_sect)
x_APB1_coords = (1064, 940, 120, p_sect)
x_APB2_coords = (1048, 1171, 140, p_sect)
x_vacuole_coords = (1000, 1648, 600, p_sect)
x_cytoplasm_coords = (1734, 1000, 300, p_sect)
# these are the vertical cuts.

y_cytoplasm_coords = (1400, 370, p_sect, 438)
y_cross_coords = (1400, 0, p_sect, img_diff.shape[0])
y_background_coords = (300, 1500, p_sect, 500)
y_vacuole1_coords = (1400, 1156, p_sect, 882)
y_vacuole2_coords = (1496, 919, p_sect, 915)
y_APB1_coords = (1117, 888, p_sect, 150)
y_APB2_coords = (1117, 1070, p_sect, 200)
y_nucleus_coords = (550, 440, p_sect, 300)

x_coords_collection = (
    x_cross_coords, x_background_coords, x_nucleus_coords, x_APB1_coords, x_APB2_coords, x_vacuole_coords,
    x_cytoplasm_coords)
y_coords_collection = (
    y_cross_coords, y_background_coords, y_nucleus_coords, y_APB1_coords, y_APB2_coords, y_vacuole1_coords,
    y_vacuole2_coords, y_cytoplasm_coords)

tmp = img_orig.copy()

# just so we  know all areas of analysis
for x_coords in x_coords_collection:
    annotate_image(x_coords, tmp, (0, 0, 0))
for y_coords in y_coords_collection:
    annotate_image(y_coords, tmp, (0, 0, 0))

# many lines
# APBs
create_avg_acf_plot(y_APB1_coords, "APB1 Y direction origin", img_orig, 'y')
create_avg_acf_plot(y_APB1_coords, "APB1 Y direction blurred", img_blurred, 'y')
create_avg_acf_plot(x_APB1_coords, "APB1 X direction origin", img_orig, 'x')
create_avg_acf_plot(x_APB1_coords, "APB1 X direction blurred", img_blurred, 'x')
create_avg_acf_plot(y_APB2_coords, "APB2 Y direction origin", img_orig, 'y')
create_avg_acf_plot(y_APB2_coords, "APB2 Y direction blurred", img_blurred, 'y')
create_avg_acf_plot(x_APB2_coords, "APB2 X direction origin", img_orig, 'x')
create_avg_acf_plot(x_APB2_coords, "APB2 X direction blurred", img_blurred, 'x')
# Nucleus
create_avg_acf_plot(y_nucleus_coords, "Nucleus Y direction origin", img_orig, 'y')
create_avg_acf_plot(y_nucleus_coords, "Nucleus Y direction blurred", img_blurred, 'y')
create_avg_acf_plot(x_nucleus_coords, "Nucleus X direction origin", img_orig, 'x')
create_avg_acf_plot(x_nucleus_coords, "Nucleus X direction blurred", img_blurred, 'x')
# Cytoplasm
create_avg_acf_plot(y_cytoplasm_coords, "Cytoplasm Y direction origin", img_orig, 'y')
create_avg_acf_plot(y_cytoplasm_coords, "Cytoplasm Y direction blurred", img_blurred, 'y')
create_avg_acf_plot(x_cytoplasm_coords, "Cytoplasm X direction origin", img_orig, 'x')
create_avg_acf_plot(x_cytoplasm_coords, "Cytoplasm X direction blurred", img_blurred, 'x')
# Background
create_avg_acf_plot(y_background_coords, "Background Y direction origin", img_orig, 'y')
create_avg_acf_plot(y_background_coords, "Background Y direction blurred", img_blurred, 'y')
create_avg_acf_plot(x_background_coords, "Background X direction origin", img_orig, 'x')
create_avg_acf_plot(x_background_coords, "Background X direction blurred", img_blurred, 'x')
# Vacuoles
create_avg_acf_plot(y_vacuole1_coords, "Vacuole1 Y direction origin", img_orig, 'y')
create_avg_acf_plot(y_vacuole1_coords, "Vacuole1 Y direction blurred", img_blurred, 'y')
create_avg_acf_plot(x_vacuole_coords, "Vacuole1 X direction origin", img_orig, 'x')
create_avg_acf_plot(x_vacuole_coords, "Vacuole1 X direction blurred", img_blurred, 'x')
create_avg_acf_plot(y_vacuole2_coords, "Vacuole2 Y direction origin", img_orig, 'y')
create_avg_acf_plot(y_vacuole2_coords, "Vacuole2 Y direction blurred", img_blurred, 'y')

# correlation of known uncorrelated objects. Cross sections
create_avg_acf_plot(y_cross_coords, "Image Y direction origin", img_orig, 'y')
create_avg_acf_plot(y_cross_coords, "Image Y direction blurred", img_blurred, 'y')
create_avg_acf_plot(x_cross_coords, "Image X direction origin", img_orig, 'x')
create_avg_acf_plot(x_cross_coords, "Image X direction blurred", img_blurred, 'x')

