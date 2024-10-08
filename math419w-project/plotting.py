import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
import statsmodels as sm
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa import stattools

def make_plots(roi_coords, img, roi_name, direction, lag_limit=100, ac_limit=0.5, control=False, uncor_img=None):
    roi_crop = create_cropped_img(img, roi_coords)
    if control:
        noise_sections = make_sections(roi_coords, uncor_img, direction)
    roi_sections = make_sections(roi_coords, img, direction)
    for section in roi_sections:
        roi_plot = pd.plotting.autocorrelation_plot(section.flatten())
    roi_plot.set_ybound(-ac_limit, ac_limit)
    roi_plot.set_xlim(0, lag_limit)
    # roi_plot.autoscale()
    roi_plot.grid()
    roi_plot.set_title(roi_name + "sections")
    plt.savefig(roi_name + ".png")
    plt.show()
    if control:
        for section in noise_sections:
            noise_plot = pd.plotting.autocorrelation_plot(section.flatten())
        noise_plot.set_ybound(-ac_limit, ac_limit)
        noise_plot.set_xlim(0, lag_limit)
        noise_plot.set_title(roi_name + "uncorrelated sections")
        plt.show()
        for section in noise_sections:
            model = AutoReg(section.flatten(), lags=10).fit()
            ar10 = model.forecast(50)
            corr_plot = pd.plotting.autocorrelation_plot(ar10)
        corr_plot.set_ybound(-ac_limit, ac_limit)
        corr_plot.set_xlim(0, lag_limit)
        corr_plot.set_title(roi_name + "correlated sections")
        plt.show()

    return roi_crop


def create_labeled_img(coords, img, color):
    x, y, w, h = coords
    labeled_img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return labeled_img


def create_cropped_img(img, coords):
    x, y, w, h = coords
    cropped_img = img[y:y + h, x: x + w]
    return cropped_img


def make_sections(roi_coords, img, direction):
    roi_crop = create_cropped_img(img, roi_coords)
    if direction == "x":
        roi_sections = np.split(roi_crop, roi_crop.shape[0])
    if direction == "y":
        roi_sections = np.hsplit(roi_crop, roi_crop.shape[1])
    return roi_sections


plt.interactive(True)
img_orig = cv2.imread('B4-058.tif', 0)

# noise reduction filter, reverse the results, basically get me the noise of the image. This one is very simple
img_blurred = cv2.GaussianBlur(img_orig, (5, 5), cv2.BORDER_DEFAULT)
img_diff = img_orig + 0.0 - img_blurred
# diff_img_viewable = (img_diff - np.min(img_diff)) / (np.max(img_diff) - np.min(img_diff)) * 255  # min max normalization
# diff_img_viewable_2 = img_diff + 127.5  # diff_image mu = 0.000, so idk if I want to transform it more.
mu = np.mean(img_diff)
sigma = np.std(img_diff)
uncor_img = np.random.normal(0, sigma, img_diff.shape)

nucleus_coords = [537, 586, 259, 274]
vacuole_coords = [1280, 1358, 560, 518]
cytoplasm_coords = [767, 211, 253, 253]
apb_coords = [1052, 1086, 142, 137]
nucleus = create_cropped_img(img_orig, nucleus_coords)
vacuole = create_cropped_img(img_orig, vacuole_coords)
cytoplasm = create_cropped_img(img_orig, cytoplasm_coords)
apb = create_cropped_img(img_orig, apb_coords)
nucleus_diff = create_cropped_img(img_diff, nucleus_coords)
vacuole_diff = create_cropped_img(img_diff, vacuole_coords)
cytoplasm_diff = create_cropped_img(img_diff, cytoplasm_coords)
apb_diff = create_cropped_img(img_diff, apb_coords)
nucleus_blur = create_cropped_img(img_blurred, nucleus_coords)
vacuole_blur = create_cropped_img(img_blurred, vacuole_coords)
cytoplasm_blur = create_cropped_img(img_blurred, cytoplasm_coords)
apb_blur = create_cropped_img(img_blurred, apb_coords)