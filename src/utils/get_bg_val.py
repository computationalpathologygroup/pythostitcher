import numpy as np
from skimage.filters import threshold_otsu
from .rgb2gray import rgb2gray


def get_bg_val(image):
    """
    Function to get the median background value

    """


    if len(image.shape) == 3:
        image = rgb2gray(image)
    elif len(image.shape) == 2:
        pass
    else:
        print(f"Expected input to be either 2D or 3D, received {len(image.shape)}D")

    thresh = threshold_otsu(image)
    bg_vals = image*(image > thresh)
    bg_vals_med = np.median(bg_vals)

    return bg_vals_med
