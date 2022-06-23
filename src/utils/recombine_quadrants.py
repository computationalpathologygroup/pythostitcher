import itertools
import numpy as np


def recombine_quadrants(images):
    """
    Custom function to recombine all transformed quadrants in one image. Images are
    recombined by computing areas of overlap and equally weighing the intensity values
    at these areas.

    Input:
    - Transformed image for all quadrants

    Output:
    - Single recombined image of all quadrants
    """

    # Initialize some parameters
    quadrants = "ABCD"
    all_combo = []
    final_im = np.zeros(images[0].shape)
    merged_mask_dict = dict()

    # Populate dicts with images and masks
    image_dict = dict()
    mask_dict = dict()

    for key, im in zip(quadrants, images):
        image_dict[key] = im
        mask_dict[key] = ((im[:, :, 0] > 0) * 1).astype("float32")

    # Find all possible quadrant combinations
    for i in range(1, len(quadrants) + 1):

        # Get all combination pairs
        combo_list = list(itertools.combinations(quadrants, i))

        # Convert tuples to strings
        if i == 1:
            combo_merged = [a[0] for a in combo_list]
        else:
            combo_merged = ["".join(item for item in a) for a in combo_list]

        all_combo += combo_merged

    # Get the mask of all combinations
    for combo in all_combo:
        num_quadrants = len(combo)
        merged_mask_dict[combo] = np.zeros(mask_dict["A"].shape)

        for item in combo:
            merged_mask_dict[combo] += mask_dict[item]

        # Get indices for a certain combo
        merged_mask_dict[combo] = merged_mask_dict[combo] == num_quadrants
        r, c = np.nonzero(merged_mask_dict[combo])

        # Get image for a certain combo
        temp = np.array([(1 / num_quadrants) * image_dict[str(ele)] for ele in combo])
        temp = np.mean(temp, axis=0)

        if len(np.squeeze(final_im.shape)) == 2:
            final_im[r, c] = temp[r, c]
        elif len(np.squeeze(final_im.shape)) == 3:
            final_im[r, c, :] = temp[r, c, :]

    final_im = final_im.astype("uint8")

    return final_im
