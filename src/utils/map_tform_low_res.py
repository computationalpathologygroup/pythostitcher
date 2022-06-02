import numpy as np
import os

from .get_resname import get_resname


def map_tform_low_res(parameters):
    """
    Custom function to upsample the previously acquired tform matrix.

    Input:
    - Dict with parameters

    Output:
    - Upsampled transformation matrix
    """

    # Calculate ratio between current resolution and previous resolution
    ratio = parameters["resolutions"][parameters['iteration']] / parameters["resolutions"][parameters['iteration'] - 1]

    # Set some filepaths for loading and saving
    prev_resname = get_resname(parameters["resolutions"][parameters['iteration']-1])
    prev_filepath_initial = f"{parameters['results_dir']}/" \
                            f"{parameters['patient_idx']}/" \
                            f"{parameters['slice_idx']}/" \
                            f"{prev_resname}/" \
                            f"tform/tform_initial.npy"
    prev_filepath_ga = f"{parameters['results_dir']}/" \
                       f"{parameters['patient_idx']}/" \
                       f"{parameters['slice_idx']}/" \
                       f"{prev_resname}/" \
                       f"tform/tform_ga.npy"
    current_resname = get_resname(parameters["resolutions"][parameters['iteration']])
    current_filepath_initial = f"{parameters['results_dir']}/" \
                               f"{parameters['patient_idx']}/" \
                               f"{parameters['slice_idx']}/" \
                               f"{current_resname}/" \
                               f"tform/tform_initial.npy"

    # Load initial tform
    if os.path.isfile(prev_filepath_initial):
        initial_tform = np.load(prev_filepath_initial, allow_pickle=True).item()
    else:
        raise ValueError(f"No initial transformation found in {prev_filepath_initial}")

    # Load genetic algorithm tform
    if os.path.isfile(prev_filepath_ga):
        ga_tform = np.load(prev_filepath_ga, allow_pickle=True).item()
    else:
        raise ValueError(f"No GA transformation found in {prev_filepath_ga}")

    new_initial_tform = dict()
    new_ga_tform = dict()
    combi_tform = dict()

    # Apply conversion ratio for all quadrants
    for quadrant in parameters["filenames"].keys():

        new_outshape = [np.round(ratio * t) for t in initial_tform[quadrant][3]]
        new_initial_tform[quadrant] = [np.round(initial_tform[quadrant][0]*ratio),
                                       np.round(initial_tform[quadrant][1]*ratio),
                                       np.round(initial_tform[quadrant][2], 1),
                                       new_outshape]
        new_ga_tform[quadrant] = [np.round(ga_tform[quadrant][0]*ratio),
                                  np.round(ga_tform[quadrant][1]*ratio),
                                  np.round(ga_tform[quadrant][2], 1),
                                  new_outshape]
        combi_tform[quadrant] = [new_initial_tform[quadrant][0] + new_ga_tform[quadrant][0],
                                 new_initial_tform[quadrant][1] + new_ga_tform[quadrant][1],
                                 new_initial_tform[quadrant][2] + new_ga_tform[quadrant][2],
                                 new_outshape]

    np.save(current_filepath_initial, combi_tform)

    return combi_tform, None

