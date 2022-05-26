import os
import numpy as np
import time
import pickle
import copy

from .get_resname import get_resname
from .genetic_algorithm import genetic_algorithm
from .map_tform_low_res import map_tform_low_res
from .plot_tools import *


def optimize_stitch(parameters, plot=False):
    """
    Function to optimize the stitching between images. This will consist of the following stages.
    1. Compute the smallest bounding box around the quadrant
    2. Compute the initial transformation by rotating this quadrant
    3. Globally align the quadrants and update the initial transformation with this translation factor
    4. Extract the edges of the quadrant and compute a Theil-Sen line through the dges
    5. Extract histograms and intensities along the edges of each quadrant.
    6. Use a genetic algorithm to optimize the initial transformation
    """

    # Make tform dir
    dirpath_tform = f"../results/{parameters['patient_idx']}/{parameters['slice_idx']}/{get_resname(parameters['resolutions'][parameters['iteration']])}/tform/"
    dirpath_quadrants = f"../results/{parameters['patient_idx']}/{parameters['slice_idx']}/{get_resname(parameters['resolutions'][parameters['iteration']])}/quadrant"
    dirpath_images = f"../results/{parameters['patient_idx']}/images"

    for path in [dirpath_tform, dirpath_images]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Check if optimized tform already exists
    filepath_tform = f"{dirpath_tform}tform_ga.npy"
    file_exists = os.path.isfile(filepath_tform)

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
            initial_tform[str(quadrant_A.quadrant_name)] = [quadrant_A.rot_trans_x+quadrant_A.crop_trans_x+quadrant_A.pad_trans_x+quadrant_A.trans_x,
                                                            quadrant_A.rot_trans_y+quadrant_A.crop_trans_y+quadrant_A.pad_trans_y+quadrant_A.trans_y,
                                                            quadrant_A.angle, quadrant_A.outshape]
            initial_tform[str(quadrant_B.quadrant_name)] = [quadrant_B.rot_trans_x+quadrant_B.crop_trans_x+quadrant_B.pad_trans_x+quadrant_B.trans_x,
                                                            quadrant_B.rot_trans_y+quadrant_B.crop_trans_y+quadrant_B.pad_trans_y+quadrant_B.trans_y,
                                                            quadrant_B.angle, quadrant_B.outshape]
            initial_tform[str(quadrant_C.quadrant_name)] = [quadrant_C.rot_trans_x+quadrant_C.crop_trans_x+quadrant_C.pad_trans_x+quadrant_C.trans_x,
                                                            quadrant_C.rot_trans_y+quadrant_C.crop_trans_y+quadrant_C.pad_trans_y+quadrant_C.trans_y,
                                                            quadrant_C.angle, quadrant_C.outshape]
            initial_tform[str(quadrant_D.quadrant_name)] = [quadrant_D.rot_trans_x+quadrant_D.crop_trans_x+quadrant_D.pad_trans_x+quadrant_D.trans_x,
                                                            quadrant_D.rot_trans_y+quadrant_D.crop_trans_y+quadrant_D.pad_trans_y+quadrant_D.trans_y,
                                                            quadrant_D.angle, quadrant_D.outshape]

            np.save(f"{dirpath_tform}/tform.npy", initial_tform)

        # If initial transformation already exists, load and upsample it.
        elif parameters['iteration'] > 0:
            initial_tform = map_tform_low_res(parameters)

        # Apply transformation to the original images
        quadrant_A.get_tformed_images(initial_tform)
        quadrant_B.get_tformed_images(initial_tform)
        quadrant_C.get_tformed_images(initial_tform)
        quadrant_D.get_tformed_images(initial_tform)
        parameters["image_centers"] = [quadrant_A.image_center, quadrant_B.image_center, quadrant_C.image_center, quadrant_D.image_center]

        # Plot transformation result
        if plot:
            plot_transformation_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters)

        # Get edges, histograms and intensities
        print(f"- extracting edges from images...")
        quadrant_A.get_edges()
        quadrant_B.get_edges()
        quadrant_C.get_edges()
        quadrant_D.get_edges()

        # Compute Theil Sen lines for edge matching
        print("- computing Theil-Sen estimation of edge...")
        quadrant_A.fit_theilsen_lines()
        quadrant_B.fit_theilsen_lines()
        quadrant_C.fit_theilsen_lines()
        quadrant_D.fit_theilsen_lines()

        # Plot all acquired Theil-Sen lines
        if plot:
            plot_theilsen_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters)

        # Compute histograms along edge
        """
        quadrant_A.get_histograms()
        quadrant_B.get_histograms()
        quadrant_C.get_histograms()
        quadrant_D.get_histograms()

        # Compute intensities along edge
        quadrant_A.get_intensities()
        quadrant_B.get_intensities()
        quadrant_C.get_intensities()
        quadrant_D.get_intensities()
        """

        # Optimization with genetic algorithm
        print("- computing reconstruction with genetic algorithm...")
        ga_dict = genetic_algorithm(quadrant_A, quadrant_B, quadrant_C, quadrant_D,
                                    parameters, initial_tform)

        # Add GA tform to initial tform
        final_tform = copy.deepcopy(initial_tform)
        final_tform["UL"][:3] = final_tform["UL"][:3] + ga_dict["best_solution"][0]
        final_tform["UR"][:3] = final_tform["UR"][:3] + ga_dict["best_solution"][1]
        final_tform["LL"][:3] = final_tform["LL"][:3] + ga_dict["best_solution"][2]
        final_tform["LR"][:3] = final_tform["LR"][:3] + ga_dict["best_solution"][3]
        np.save(filepath_tform, final_tform)

        # Plot the results of the genetic algorithm
        plot_ga_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, final_tform, ga_dict)

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
