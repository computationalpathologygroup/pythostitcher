from skimage import segmentation
import numpy as np


def get_boundary(mask):
    """
    Function to get the boundary pixels of a mask

    :param mask: input image
    :return: boundary pixels
    """

    assert len(mask.shape) == 2, f"input must be a 2D mask, detected {len(mask.shape)}D"
    assert len(np.unique(mask)) == 2, "input must be a binary mask"

    boundary = segmentation.find_boundaries(mask, mode="outer")

    return c, r