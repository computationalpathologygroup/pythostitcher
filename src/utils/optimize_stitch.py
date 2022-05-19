import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

from .get_resname import get_resname
from .auto_rotate import auto_rotate
from .crop_img import crop_img
from .get_tformed_images import get_tformed_images
from .stitch_imfuse import stitch_imfuse
from .get_edges import get_edges
from .packup import packup_edges, packup_images
from .spatial_ref_object import spatial_ref_object
from .get_theilsen_lines import get_theilsen_lines
from .get_histograms_and_intensities import get_histograms_and_intensities
from .genetic_algorithm import genetic_algorithm
from .get_filename import get_filename
from .map_tform_low_res import map_tform_low_res
from .verify_overlap import verify_non_overlap
from .plot_tools import *



def optimize_stitch(parameters, plot=False):
    """
    Function to optimize the stitching between images
    """

    # Make tform dir
    dirpath_tform = f"../results/{parameters['patient_idx']}/{parameters['slice_idx']}/{get_resname(parameters['resolutions'][parameters['iteration']])}/tform/"
    dirpath_quadrants = f"../results/{parameters['patient_idx']}/{parameters['slice_idx']}/{get_resname(parameters['resolutions'][parameters['iteration']])}/quadrant"

    if not os.path.exists(dirpath_tform):
        os.mkdir(dirpath_tform)

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

        """
        # Create spatial reference object (similar to matlab's imref2d)
        quadrant_A.ref_object = spatial_ref_object(quadrant_A.gray_image)
        quadrant_B.ref_object = spatial_ref_object(quadrant_B.gray_image)
        quadrant_C.ref_object = spatial_ref_object(quadrant_C.gray_image)
        quadrant_D.ref_object = spatial_ref_object(quadrant_D.gray_image)
        """

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

            # Save transformation for all quadrants.
            tform = dict()
            tform[str(quadrant_A.quadrant_name)] = [quadrant_A.rot_trans_x, quadrant_A.rot_trans_y,
                                                    quadrant_A.crop_trans_x, quadrant_A.crop_trans_y,
                                                    quadrant_A.pad_trans_x, quadrant_A.pad_trans_y,
                                                    quadrant_A.trans_x, quadrant_A.trans_y,
                                                    quadrant_A.angle, quadrant_A.outshape]
            tform[str(quadrant_B.quadrant_name)] = [quadrant_B.rot_trans_x, quadrant_B.rot_trans_y,
                                                    quadrant_B.crop_trans_x, quadrant_B.crop_trans_y,
                                                    quadrant_B.pad_trans_x, quadrant_B.pad_trans_y,
                                                    quadrant_B.trans_x, quadrant_B.trans_y,
                                                    quadrant_B.angle, quadrant_B.outshape]
            tform[str(quadrant_C.quadrant_name)] = [quadrant_C.rot_trans_x, quadrant_C.rot_trans_y,
                                                    quadrant_C.crop_trans_x, quadrant_C.crop_trans_y,
                                                    quadrant_C.pad_trans_x, quadrant_C.pad_trans_y,
                                                    quadrant_C.trans_x, quadrant_C.trans_y,
                                                    quadrant_C.angle, quadrant_C.outshape]
            tform[str(quadrant_D.quadrant_name)] = [quadrant_D.rot_trans_x, quadrant_D.rot_trans_y,
                                                    quadrant_D.crop_trans_x, quadrant_D.crop_trans_y,
                                                    quadrant_D.pad_trans_x, quadrant_D.pad_trans_y,
                                                    quadrant_D.trans_x, quadrant_D.trans_y,
                                                    quadrant_D.angle, quadrant_D.outshape]

            np.save(f"{dirpath_tform}/tform.npy", tform)

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

        # If initial transformation already exists, load and upsample it.
        elif parameters['iteration'] > 0:
            initial_tform = map_tform_low_res(parameters)

        # Apply transformation to the original images
        quadrant_A.get_tformed_images(initial_tform)
        quadrant_B.get_tformed_images(initial_tform)
        quadrant_C.get_tformed_images(initial_tform)
        quadrant_D.get_tformed_images(initial_tform)
        parameters["image_centers"] = [quadrant_A.image_center, quadrant_B.image_center,
                                       quadrant_C.image_center, quadrant_D.image_center]

        # Plot tformed bbox cornerpoints
        """
        if plot and parameters['iteration']==0:
            plot_rotated_bbox(quadrant_A, quadrant_B, quadrant_C, quadrant_D)
        """

        # Plot transformation result
        if plot:
            plot_transformation_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters)

        # Verify that there are no overlapping segments
        verify_non_overlap(quadrant_A, quadrant_B, quadrant_C, quadrant_D, tolerance=10)

        # Get edges, histograms and intensities
        print(f"- extracting edges from images...")
        quadrant_A.get_edges(plot=False)
        quadrant_B.get_edges(plot=False)
        quadrant_C.get_edges(plot=False)
        quadrant_D.get_edges(plot=False)

        # Compute Theil Sen lines for edge matching
        print("- computing Theil-Sen estimation of edge...")
        quadrant_A.fit_theilsen_lines(plot=False)
        quadrant_B.fit_theilsen_lines(plot=False)
        quadrant_C.fit_theilsen_lines(plot=False)
        quadrant_D.fit_theilsen_lines(plot=False)

        # Plot all acquired Theil-Sen lines
        if plot:
            plot_theilsen_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters)

        # Compute histograms along edge
        quadrant_A.get_histograms()
        quadrant_B.get_histograms()
        quadrant_C.get_histograms()
        quadrant_D.get_histograms()

        # Compute intensities along edge
        quadrant_A.get_intensities()
        quadrant_B.get_intensities()
        quadrant_C.get_intensities()
        quadrant_D.get_intensities()

        #plot_tformed_lines(quadrant_A, quadrant_B, quadrant_C, quadrant_D)


        # Optimization with genetic algorithm
        print("- computing reconstruction with genetic algorithm...")
        solution, solution_fitness = genetic_algorithm(quadrant_A, quadrant_B, quadrant_C, quadrant_D,
                                                       parameters, initial_tform, plot=True)


        """
        # Compute total transformation
        tform = initial_tform
        tform[3:] = tform[3:] + solution

        # Save results
        ga_dict = dict()
        ga_dict["solution"] = solution
        ga_dict["solution_fitness"] = solution_fitness
        ga_dict["tform"] = tform
        np.save(filepath_tform, ga_dict)
        """

        # Provide verbose on computation time
        end_time = time.time()
        current_res = parameters["resolutions"][parameters["iteration"]]
        sec = np.round(end_time-start_time, 1)
        print(f"> time to optimize patient {parameters['patient_idx']} at resolution {current_res}: {sec} seconds")

        """

    else:
        GA_dict = np.load(filepath_tform, allow_pickle=True).item()
        solution_fitness = GA_dict["solution_fitness"]
        
        """

    return None
