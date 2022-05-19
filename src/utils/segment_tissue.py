import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure, morphology, segmentation, io, transform
import multiresolutionimageinterface as mir
from PIL import Image


def get_optim_mask_level(mask, value):
    """
    Function to return the optimal mask level defined by closeness to a given threshold

    input:
    - mask (imagereader object)
    - required matrix size, e.g. the function will interpret 4000 as
      searching for the closest dimensions to a (4000, 4000) matrix (int)
    """

    # Get list of dimensions
    mask_dims = sorted([mask.getLevelDimensions(i) for i in range(mask.getNumberOfLevels())])

    # Calculate SSD for levels in mask
    optim = [((i[0] - value) ** 2 + (i[1] - value) ** 2) for i in mask_dims]

    # Get best index
    idx = np.argmin(optim)

    # Compensate for sorted dims list
    best_level = mask.getNumberOfLevels() - idx - 1

    return best_level


def get_largest_cc(mask):
    """
    Function to get the largest connect component in an image using skimage.

    input:
    - mask obtained from segmentation algorithm (numpy.array)

    output:
    - mask containing only the largest connected component (numpy.array)
    """

    # Label different components in image
    mask_label = measure.label(mask)

    # Compute region properties
    props = measure.regionprops(mask_label)

    # Get area of each component
    areas = [props[i]["area"] for i in range(len(props))]

    # Get index of largest area
    idx = np.argmax(areas)

    # Get label of largest area
    label = int(props[idx]["label"])

    # Get largest connected component
    largest_cc = (mask_label == label) * 1

    return largest_cc


def postprocess_largest_cc(mask):
    """
    Function that performs post processing on the largest component and returns
    a bounding box that captures the final mask.

    input:
    - mask of largest connected component (numpy.array)

    output:
    - processed mask (numpy.array)
    """

    # Compute max hole size. 20% of total sum is arbitrary
    hole_size = int(np.round(0.5 * np.sum(mask)))

    # Remove small holes
    mask = morphology.remove_small_holes(mask, hole_size)

    # Perform closing for smoothing border. Disk size is arbitrary
    kernel = morphology.disk(20)
    mask = morphology.binary_closing(mask, kernel) * 1

    # Empirical tests show that some holes may persist. Perform operation again
    mask = morphology.remove_small_holes(mask, hole_size) * 1

    # Get bounding box around mask
    props = measure.regionprops(mask)
    bbox = props[0]["bbox"]

    # Change bbox values to include a small zero border around bbox
    pad = 50
    bbox_new = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]

    return mask, bbox_new


def segment_tissue(impath, img):
    """
    Function that segments the tissue from the image.

    FOR NOW THIS FUNCTION WILL JUST LOAD THE PREPROCESSED MASKS
    """

    # Navigate to mask path and load mask
    mask_path = impath.replace("images", "masks")
    mask = np.array(io.imread(mask_path))

    # Mask is false RGB because of TIF requirements.
    mask = mask[:, :, 0]

    # Resize to same shape as input image
    new_dims = img.shape

    if len(new_dims) == 2:
        mask = transform.resize(mask, new_dims)
    elif len(new_dims) == 3:
        mask = transform.resize(mask, new_dims[:-1])
    else:
        print(f"Expected input image to be 2D or 3D, received input {len(new_dims)}D")


    return mask


