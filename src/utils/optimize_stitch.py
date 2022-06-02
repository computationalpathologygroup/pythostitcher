import os
import numpy as np
import time
import pickle
import copy

from .get_resname import get_resname
from .genetic_algorithm import genetic_algorithm_global, genetic_algorithm_piecewise
from .map_tform_low_res import map_tform_low_res
from .plot_tools import *


def optimize_stitch(parameters, assembly='global', plot=False):
    """
    Function to optimize the stitching between images. This will consist of the following steps.
    1. Compute the smallest bounding box around the quadrant
    2. Compute the initial transformation by rotating this quadrant
    3. Globally align the quadrants and update the initial transformation with this translation factor
    4. Extract the edges of the quadrant and compute a Theil-Sen line through the edges
    5. Use a genetic algorithm to optimize the initial transformation

    Input:
    - Dictionary with parameters
    - Assembly method (global | piecewise). This parameter indicates whether the stitching should be performed
    for all quadrants at once or piecewise quadrant by quadrant.
    - Boolean value whether to plot outcomes

    Output:
    - Final stitched image
    """

    # Make some directories for saving results
    dirpath_tform = f"../results/" \
                    f"{parameters['patient_idx']}/" \
                    f"{parameters['slice_idx']}/" \
                    f"{get_resname(parameters['resolutions'][parameters['iteration']])}/" \
                    f"tform/"
    dirpath_images = f"../results/" \
                     f"{parameters['patient_idx']}/" \
                     f"images"
    dirpath_quadrants = f"../results/" \
                        f"{parameters['patient_idx']}/" \
                        f"{parameters['slice_idx']}/" \
                        f"{get_resname(parameters['resolutions'][parameters['iteration']])}/" \
                        f"quadrant"

    for path in [dirpath_tform, dirpath_images]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Check if optimized tform already exists
    parameters["filepath_tform"] = f"{dirpath_tform}tform_ga.npy"
    file_exists = os.path.isfile(parameters["filepath_tform"])

    # Start optimizing stitch
    if not file_exists:
        start_time = time.time()

        # Load previously saved quadrants
        with open(f"{dirpath_quadrants}_UL", "rb") as loadfile:
            quadrant_A = pickle.load(loadfile)
        with open(f"{dirpath_quadrants}_UR", "rb") as loadfile:
            quadrant_B = pickle.load(loadfile)
        with open(f"{dirpath_quadrants}_LL", "rb") as loadfile:
            quadrant_C = pickle.load(loadfile)
        with open(f"{dirpath_quadrants}_LR", "rb") as loadfile:
            quadrant_D = pickle.load(loadfile)

        # Load images
        print("- loading images...")
        quadrant_A.load_images()
        quadrant_B.load_images()
        quadrant_C.load_images()
        quadrant_D.load_images()

        # Perform initialization at the lowest resolution
        if parameters['iteration'] == 0:

            # Get bounding box based on the tissue mask
            quadrant_A.get_bbox_corners(quadrant_A.mask)
            quadrant_B.get_bbox_corners(quadrant_B.mask)
            quadrant_C.get_bbox_corners(quadrant_C.mask)
            quadrant_D.get_bbox_corners(quadrant_D.mask)

            # Get the initial transform consisting of rotation and cropping
            quadrant_A.get_initial_transform()
            quadrant_B.get_initial_transform()
            quadrant_C.get_initial_transform()
            quadrant_D.get_initial_transform()

            # Plot the rotation result as a visual check
            if plot:
                plot_rotation_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D)

            # Compute local transformation to align horizontal pieces. Input for this method is
            # the horizontal neighbour of this quadrant
            quadrant_A.get_tformed_images_local(quadrant_B)
            quadrant_B.get_tformed_images_local(quadrant_A)
            quadrant_C.get_tformed_images_local(quadrant_D)
            quadrant_D.get_tformed_images_local(quadrant_C)

            # Compute global transformation to align all pieces. Input for this method is one of
            # the vertical neighbours of this quadrant
            quadrant_A.get_tformed_images_global(quadrant_C)
            quadrant_B.get_tformed_images_global(quadrant_C)
            quadrant_C.get_tformed_images_global(quadrant_A)
            quadrant_D.get_tformed_images_global(quadrant_A)

            # Get final tform params for plotting later on
            initial_tform = dict()
            A_total_x = quadrant_A.rot_trans_x+quadrant_A.crop_trans_x+quadrant_A.pad_trans_x+quadrant_A.trans_x
            A_total_y = quadrant_A.rot_trans_y+quadrant_A.crop_trans_y+quadrant_A.pad_trans_y+quadrant_A.trans_y
            initial_tform[str(quadrant_A.quadrant_name)] = [A_total_x, A_total_y, quadrant_A.angle, quadrant_A.outshape]

            B_total_x = quadrant_B.rot_trans_x + quadrant_B.crop_trans_x + quadrant_B.pad_trans_x + quadrant_B.trans_x
            B_total_y = quadrant_B.rot_trans_y + quadrant_B.crop_trans_y + quadrant_B.pad_trans_y + quadrant_B.trans_y
            initial_tform[str(quadrant_B.quadrant_name)] = [B_total_x, B_total_y,  quadrant_B.angle, quadrant_B.outshape]

            C_total_x = quadrant_C.rot_trans_x + quadrant_C.crop_trans_x + quadrant_C.pad_trans_x + quadrant_C.trans_x
            C_total_y = quadrant_C.rot_trans_y + quadrant_C.crop_trans_y + quadrant_C.pad_trans_y + quadrant_C.trans_y
            initial_tform[str(quadrant_C.quadrant_name)] = [C_total_x, C_total_y, quadrant_C.angle, quadrant_C.outshape]

            D_total_x = quadrant_D.rot_trans_x + quadrant_D.crop_trans_x + quadrant_D.pad_trans_x + quadrant_D.trans_x
            D_total_y = quadrant_D.rot_trans_y + quadrant_D.crop_trans_y + quadrant_D.pad_trans_y + quadrant_D.trans_y
            initial_tform[str(quadrant_D.quadrant_name)] = [D_total_x, D_total_y, quadrant_D.angle, quadrant_D.outshape]

            np.save(f"{dirpath_tform}/tform_initial.npy", initial_tform)
            ga_tform = None

        # If initial transformation already exists, load and upsample it.
        elif parameters['iteration'] > 0:
            initial_tform, ga_tform = map_tform_low_res(parameters)

        # Apply transformation to the original images
        quadrant_A.get_tformed_images(initial_tform=initial_tform, ga_tform=ga_tform)
        quadrant_B.get_tformed_images(initial_tform=initial_tform, ga_tform=ga_tform)
        quadrant_C.get_tformed_images(initial_tform=initial_tform, ga_tform=ga_tform)
        quadrant_D.get_tformed_images(initial_tform=initial_tform, ga_tform=ga_tform)
        parameters["image_centers"] = [quadrant_A.image_center, quadrant_B.image_center,
                                       quadrant_C.image_center, quadrant_D.image_center]

        # Plot transformation result
        if plot:
            plot_transformation_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters)

        # Get edges from quadrants
        print(f"- extracting edges from images...")
        quadrant_A.get_edges()
        quadrant_B.get_edges()
        quadrant_C.get_edges()
        quadrant_D.get_edges()

        # Compute Theil Sen lines through edges
        print("- computing Theil-Sen estimation of edge...")
        quadrant_A.fit_theilsen_lines()
        quadrant_B.fit_theilsen_lines()
        quadrant_C.fit_theilsen_lines()
        quadrant_D.fit_theilsen_lines()

        # Plot all acquired Theil-Sen lines
        if plot:
            plot_theilsen_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters)

        # Optimization with genetic algorithm
        print("- computing reconstruction with genetic algorithm...")
        if assembly == "global":
            ga_dict = genetic_algorithm_global(quadrant_A, quadrant_B, quadrant_C, quadrant_D,
                                               parameters, initial_tform)
        elif assembly == "piecewise":
            ga_dict = genetic_algorithm_piecewise(quadrant_A, quadrant_B, quadrant_C, quadrant_D,
                                                  parameters, initial_tform)

        # Plot the results of the genetic algorithm
        plot_ga_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, ga_dict)

        # Provide verbose on computation time
        end_time = time.time()
        current_res = parameters["resolutions"][parameters["iteration"]]
        sec = np.round(end_time-start_time, 1)
        print(f"> time to optimize patient {parameters['patient_idx']} at resolution {current_res}: {sec} seconds")

    else:
        print("- already optimized this resolution!")

    # At final resolution provide some extra visualizations
    if parameters["iteration"] == 3:

        # Make a gif of the tform result
        make_tform_gif(parameters)

        # Plot the fitness trajectory over the multiple resolutions
        plot_ga_multires(parameters)

    return
