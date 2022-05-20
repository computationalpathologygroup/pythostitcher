import numpy as np
import os

from .get_resname import get_resname


def map_tform_low_res(parameters):
    """
    Custom function to upsample the previously acquired tform matrix.
    """

    # Calculate ratio between current resolution and previous resolution
    ratio = parameters["resolutions"][parameters['iteration']] / parameters["resolutions"][parameters['iteration'] - 1]

    # Load tform that was optimized with GA in previous resolution
    prev_resname = get_resname(parameters["resolutions"][parameters['iteration']-1])
    prev_filepath = f"{parameters['results_dir']}/{parameters['patient_idx']}/{parameters['slice_idx']}/{prev_resname}/tform/tform_ga.npy"

    if os.path.isfile(prev_filepath):
        tform = np.load(prev_filepath, allow_pickle=True).item()
    else:
        raise ValueError(f"No GA transformation found in {prev_filepath}")

    new_tform = dict()

    # Apply conversion ratio for all quadrants
    for quadrant in parameters["filenames"].keys():

        new_outshape = [np.round(ratio * t) for t in tform[quadrant][3]]
        new_tform[quadrant] = [tform[quadrant][0]*ratio, tform[quadrant][1]*ratio,
                               tform[quadrant][2], new_outshape]

    return new_tform


