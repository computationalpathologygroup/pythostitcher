import numpy as np


def apply_im_tform_to_coords(coords, fragment, rot_k):
    """
    Convenience function to apply a 90 degree image rotation to coordinates. You could
    of course do this through coordinate transform, but this is overly complex due to
    changing centers of rotation and image shifts. This function just converts the coords
    to a binary image, rotates the image and extracts the coords.

    NOTE: this function will also scale the coordinates to the same range as the
    final output image.
    """

    # Downscale for computational efficiency
    downscale = np.ceil(fragment.width/2000).astype("int")

    # Downscale coords for efficiency
    coords_ds = (coords / downscale).astype("int")

    # Clip coords to prevent out of bounds indexing due to rounding errors
    coords_image_dims = (int(fragment.width / downscale),
                        int(fragment.height / downscale))

    coords_ds_x = coords_ds[:, 0]
    coords_ds_y = coords_ds[:, 1]

    # Convert coords to image. Use larger image to prevent out of bounds indexing.
    offset = int(0.3 * (coords_image_dims[0]))
    coords_image = np.zeros((coords_image_dims[0] + 2 * offset, coords_image_dims[1] + 2 * offset))
    coords_image[coords_ds_x + offset, coords_ds_y + offset] = 1

    # Sanity checks
    assert len(coords_ds_x) == len(coords_ds_y), \
        "mismatch in number of x/y coordinates, check input coordinates"
    assert len(coords_ds_x) == np.sum(coords_image).astype("int"), \
        "received duplicate coordinates, check input coordinates"

    # Rot image and extract coords
    coords_image = np.rot90(coords_image, rot_k, (0, 1))
    r, c = np.nonzero(coords_image)
    coords_image_rot = np.vstack([r - offset, c - offset]).T
    coords_image_rot = (coords_image_rot * downscale).astype("int")

    # Sort coords by x or y values depending on line direction
    if np.std(coords_image_rot[:, 0]) > np.std(coords_image_rot[:, 1]):
        coords_image_rot_sort = sorted(coords_image_rot, key=lambda x: x[0])
        coords_image_rot_sort = np.array(coords_image_rot_sort)
    else:
        coords_image_rot_sort = sorted(coords_image_rot, key=lambda x: x[1])
        coords_image_rot_sort = np.array(coords_image_rot_sort)

    return coords_image_rot_sort
