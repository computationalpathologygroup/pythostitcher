import numpy as np


def crop_img(img):
    """
    Custom function to crop image to only include nonzero pixels

    Input:
    - image (np.array)

    Output:
    - cropped image (np.array)
    - cropping indices (list)

    """

    assert len(img.shape) == 2, "input must be 2D"

    # Get cropping indices
    r, c = np.nonzero(img)
    minR = np.min(r)
    maxR = np.max(r)
    minC = np.min(c)
    maxC = np.max(c)
    width = np.abs(maxC-minC)
    height = np.abs(maxR-minR)

    # Slice the array
    img_crop = img[minC:minC+width, minR:minR+height]

    # if nargout == 1:
    #     varargout = img_cropped
    # elif nargout == 2:
    #     varargout = [0]*2
    #     varargout[0] = img_cropped
    #     varargout[1] = [minR, maxR, minC, maxC]
    # else:
    #     raise ValueError("Unexpected number of output arguments")

    return img_crop, [minR, maxR, minC, maxC]
