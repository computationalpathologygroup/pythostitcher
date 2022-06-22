import numpy as np
import os

from .get_resname import get_resname


def map_tform_low_res(parameters):
    """
    Custom function to upsample the previously acquired tform matrix. This upsampling can be performed linearly
    since the different images from different resolutions were preprocessed to be the same ratio relative to the
    change in resolution.

    Input:
    - Dict with parameters

    Output:
    - Upsampled transformation matrix
    """

    # Calculate ratio between current resolution and previous resolution
    ratio = parameters["resolutions"][parameters['iteration']] / parameters["resolutions"][parameters['iteration'] - 1]

    # Set some filepaths for loading and saving
    prev_resname = get_resname(parameters["resolutions"][parameters['iteration']-1])
    prev_filepath_final_tform = f"{parameters['results_dir']}/" \
                                f"{parameters['patient_idx']}/" \
                                f"tform/" \
                                f"{prev_resname}_tform_final.npy"

    current_resname = get_resname(parameters["resolutions"][parameters['iteration']])
    current_filepath_initial_tform = f"{parameters['results_dir']}/" \
                                     f"{parameters['patient_idx']}/" \
                                     f"tform/" \
                                     f"{current_resname}_tform_initial.npy" \

    # Load genetic algorithm tform
    if os.path.isfile(prev_filepath_final_tform):
        initial_tform = np.load(prev_filepath_final_tform, allow_pickle=True).item()
    else:
        raise ValueError(f"No transformation found in {prev_filepath_final_tform}")

    new_initial_tform = dict()

    # Apply conversion ratio for all quadrants. Each transformation matrix is organised as follows:
    # [translation_x (int), translation_y (int), angle (float), center to rotate around (tuple), output_shape (tuple)]
    for quadrant in parameters["filenames"].keys():

        new_center = [int(np.round(ratio * t)) for t in initial_tform[quadrant][3]]
        new_center = tuple(new_center)
        new_outshape = [int(np.round(ratio * t)) for t in initial_tform[quadrant][4]]
        new_outshape = tuple(new_outshape)

        new_initial_tform[quadrant] = [int(np.round(initial_tform[quadrant][0]*ratio)),
                                       int(np.round(initial_tform[quadrant][1]*ratio)),
                                       np.round(initial_tform[quadrant][2], 1),
                                       new_center, new_outshape]

    # Although we technically don't need to save this initial tform for this resolution, this can come in handy for
    # comparing the results of the genetic algorithm for a given resolution.
    np.save(current_filepath_initial_tform, new_initial_tform)

    return new_initial_tform
