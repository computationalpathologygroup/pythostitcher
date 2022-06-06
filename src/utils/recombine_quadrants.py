import numpy as np
import itertools


def recombine_quadrants(im_A, im_B, im_C, im_D):
    """
    Custom function to recombine all transformed quadrants in one image. Images are recombined by computing areas
    of overlap and equally weighing the intensity values at these areas.

    Input:
    - Transformed image for all quadrants

    Output:
    - Single recombined image of all quadrants
    """

    # Initialize some parameters
    quadrants = "ABCD"
    image_dict = dict()
    image_dict["A"], image_dict["B"], image_dict["C"], image_dict["D"] = im_A, im_B, im_C, im_D
    all_combo = []
    final_im = np.zeros((im_A.shape))

    # Obtain masks
    if len(np.squeeze(im_A.shape))==2:
        mask_A, mask_B, mask_C, mask_D = im_A>0, im_B>0, im_C>0, im_D>0
    elif len(np.squeeze(im_A.shape))==3:
        mask_A, mask_B, mask_C, mask_D = im_A[:, :, 0]>0, im_B[:, :, 0]>0, im_C[:, :, 0]>0, im_D[:, :, 0]>0
    else:
        assert ValueError(f"Expected input image to be either 2D/3D, received {len(np.squeeze(im_A.shape))}D")

    mask_dict = dict()
    mask_dict["A"], mask_dict["B"], mask_dict["C"], mask_dict["D"] = mask_A, mask_B, mask_C, mask_D
    merged_mask_dict = dict()

    # Find all possible quadrant combinations
    for i in range(1, len(quadrants)+1):

        # Get all combination pairs
        combo_list = list(itertools.combinations(quadrants, i))

        # Convert tuples to strings
        if i==1:
            combo_merged = [a[0] for a in combo_list]
        else:
            combo_merged = ["".join(item for item in a) for a in combo_list]

        all_combo += combo_merged

    # Get the mask of all combinations
    for combo in all_combo:
        num_quadrants = len(combo)
        merged_mask_dict[combo] = np.zeros((mask_A.shape))

        for item in combo:
            merged_mask_dict[combo] += mask_dict[item]

        # Get indices for a certain combo
        merged_mask_dict[combo] = merged_mask_dict[combo]==num_quadrants
        r, c = np.nonzero(merged_mask_dict[combo])

        # Get image for a certain combo
        temp = np.array([(1/num_quadrants) * image_dict[str(ele)] for ele in combo])
        temp = np.mean(temp, axis=0)

        if len(np.squeeze(final_im.shape))==2:
            final_im[r, c] = temp[r, c]
        elif len(np.squeeze(final_im.shape))==3:
            final_im[r, c, :] = temp[r, c, :]

    final_im = final_im.astype("uint8")

    return final_im
