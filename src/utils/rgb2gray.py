
def rgb2gray(im):
    """
    Function to convert RGB images to grayscale

    input: image with rgb channels
    out:grayscale image
    """

    assert (len(im.shape) == 3) and (im.shape[-1] == 3), "input must have 3 channels"

    out = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]

    return out
