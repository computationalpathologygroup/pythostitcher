import numpy as np
import cv2


def warp_2d_points(src, center, rotation, translation):
    """
    Custom function to warp a set of 2D coordinates using an affine transform.
    """

    # Catch use case where only 1 coordinate pair is provided as input
    if len(np.array(src).shape) == 1:
        src = np.array(src)
        src = np.transpose(src[:, np.newaxis])

    assert len(np.array(src).shape) == 2 and np.array(src).shape[-1] == 2, "Input must be 2 dimensionsal and be ordered as Nx2 matrix"
    assert len(translation) == 2, "Translation must consist of X/Y component"

    # Ensure variables are in correct format
    center = tuple([int(i) for i in np.squeeze(center)])
    src = src.astype("float32")

    # Create rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center=center, angle=rotation, scale=1)
    rot_mat[0, 2] += translation[0]
    rot_mat[1, 2] += translation[1]

    # Add list of ones as pseudo third dimension to ensure proper matrix calculations
    add_ones = np.ones((src.shape[0], 1))
    src = np.hstack([src, add_ones])

    # Transform points
    tform_src = rot_mat.dot(src.T).T
    tform_src = np.round(tform_src, 1)

    return tform_src


def warp_image(src, center, rotation, translation, output_shape=None):
    """
    Custom function to warp a 2D image using an affine transformation.
    """

    # Ensure that shape only holds integers
    output_shape = [int(i) for i in output_shape]

    # Get output shape if it is specified. Switch XY for opencv convention
    if output_shape:
        if len(output_shape) == 2:
            output_shape = tuple(output_shape[::-1])
        elif len(output_shape) == 3:
            output_shape = tuple(output_shape[:2][::-1])
    # Else keep same output size as input image
    else:
        if len(src.shape) == 2:
            output_shape = src.shape
            output_shape = tuple(output_shape[::-1])
        elif len(src.shape) == 3:
            output_shape = src.shape[:2]
            output_shape = tuple(output_shape[::-1])

    # Convert to uint8 for opencv
    if src.dtype == "float32":
        src = ((src/np.max(src))*255).astype("uint8")

    # Ensure center is in correct format
    center = tuple([int(i) for i in np.squeeze(center)])

    # Create rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center=center, angle=rotation, scale=1)
    rot_mat[0, 2] += translation[0]
    rot_mat[1, 2] += translation[1]

    # Warp image
    tform_src = cv2.warpAffine(src, rot_mat, output_shape)

    return tform_src
