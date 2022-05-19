import numpy as np
import os

from .get_resname import get_resname


def map_tform_low_res(parameters):
    """
    Custom function to upsample up the tform based on the tform from previous resolution
    """

    # Calculate ratio between resolutions
    ratio = parameters["resolutions"][parameters['iteration']]/parameters["resolutions"][parameters['iteration']-1]

    # Get previous tform
    prev_resname = get_resname(parameters["resolutions"][parameters['iteration']-1])
    prev_filepath = f"{parameters['results_dir']}/{parameters['patient_idx']}/{parameters['slice_idx']}/{prev_resname}/tform/tform.npy"

    if os.path.isfile(prev_filepath):
        tform = np.load(prev_filepath, allow_pickle=True).item()
    else:
        raise ValueError(f"No transformation found in {prev_filepath}")

    # Create new dict for new tform
    new_tform = dict()
    save_tform = dict()

    # Apply conversion ratio for all quadrants
    for quadrant in parameters["filenames"].keys():

        # Compute new translation values (crop + actual) and output shape
        new_trans_x = ratio*tform[quadrant][0] + ratio*tform[quadrant][2] + ratio*tform[quadrant][4] + ratio*tform[quadrant][6]
        new_trans_y = ratio*tform[quadrant][1] + ratio*tform[quadrant][3] + ratio*tform[quadrant][5] + ratio*tform[quadrant][7]
        new_outshape = [ratio*t for t in tform[quadrant][9]]
        new_tform[quadrant] = [new_trans_x, new_trans_y, tform[quadrant][8], new_outshape]

        # Save all tform values for next iteration. Only angle should not be scaled.
        save_tform[quadrant] = [ratio*tform[quadrant][0], ratio*tform[quadrant][1], ratio*tform[quadrant][2],
                                ratio*tform[quadrant][3], ratio*tform[quadrant][4], ratio*tform[quadrant][5],
                                ratio*tform[quadrant][6], ratio*tform[quadrant][7], tform[quadrant][8],
                                [ratio*a for a in tform[quadrant][9]]]

    # Save current values
    current_resname = get_resname(parameters["resolutions"][parameters['iteration']])
    current_filepath = f"{parameters['results_dir']}/{parameters['patient_idx']}/{parameters['slice_idx']}/{current_resname}/tform/tform.npy"
    np.save(current_filepath, save_tform)

    return new_tform
