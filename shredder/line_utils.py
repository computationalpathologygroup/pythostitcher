import numpy as np


def interpolate_contour(contour):
    """
    Function to interpolate a contour which is represented by a set of points.
    Example:
    contour = [[0, 1], [1, 5], [2, 10]]
    new_contour = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [1, 6] etc.]
    """

    assert type(contour) == np.ndarray, "contour must be of type numpy array"
    assert len(contour.shape) == 2, "contour must be 2-dimensional"

    for i in range(len(contour) - 1):

        # Get x and y values to interpolate on
        xvals = np.array([contour[i, 0], contour[i + 1, 0]]).astype("int")
        yvals = np.array([contour[i, 1], contour[i + 1, 1]]).astype("int")

        # Create steps of size 1
        max_dif = np.max([np.abs(xvals[1] - xvals[0]), np.abs(yvals[1] - yvals[0])])
        new_xvals = np.linspace(xvals[0], xvals[1], num=max_dif).astype("int")
        new_yvals = np.linspace(yvals[0], yvals[1], num=max_dif).astype("int")

        # Get interpolated contour
        interp_contour = np.array([new_xvals, new_yvals]).T

        # Add interpolated values to new contour
        if i == 0:
            new_contour = interp_contour
        else:
            new_contour = np.vstack([new_contour, interp_contour])

    return new_contour


def apply_im_tform_to_coords(coords, fragment, downscale, rot_k):
    """
    Convenience function to apply a 90 degree image rotation to coordinates. You could
    of course do this through coordinate transform, but this is overly complex due to
    changing centers of rotation and image shifts. This function just converts the coords
    to a binary image, rotates the image and extracts the coords.
    """

    # Downscale coords for efficiency
    coords_ds = (coords / downscale).astype("int")

    # Clip coords to prevent out of bounds indexing due to rounding errors
    coords_image_dims = (int(fragment.width / downscale),
                        int(fragment.height / downscale))
    coords_ds_x = np.clip(coords_ds[:, 0], 0, coords_image_dims[0]-1)
    coords_ds_y = np.clip(coords_ds[:, 1], 0, coords_image_dims[1]-1)

    # Convert to image
    coords_image = np.zeros((coords_image_dims))
    coords_image[coords_ds_x, coords_ds_y] = 1

    # Rot image and extract coords
    coords_image = np.rot90(coords_image, rot_k, (0, 1))
    r, c = np.nonzero(coords_image)
    coords_image_rot = np.vstack([r, c]).T
    coords_image_rot = (coords_image_rot * downscale).astype("int")

    # Sort coords by x or y values depending on line direction
    if np.std(coords_ds[:, 0]) > np.std(coords_ds[:, 1]):
        coords_image_rot_sort = sorted(coords_image_rot, key=lambda x: x[0])
        coords_image_rot_sort = np.array(coords_image_rot_sort)
    else:
        coords_image_rot_sort = sorted(coords_image_rot, key=lambda x: x[1])
        coords_image_rot_sort = np.array(coords_image_rot_sort)

    return coords_image_rot_sort
