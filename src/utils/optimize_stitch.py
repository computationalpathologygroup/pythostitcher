import time
import pickle

from .get_resname import get_resname
from .genetic_algorithm import genetic_algorithm
from .map_tform_low_res import map_tform_low_res
from .plot_tools import *


def optimize_stitch(parameters):
    """
    Function to optimize the stitching between quadrants. This will consist of the
    following steps:
    1. Compute the smallest bounding box around the quadrant
    2. Rotate the quadrant as a first step towards alignment
    3. Globally align the quadrants such that they share the same coordinate system
    4. Identify cornerpoints in the quadrant and extract relevant edges
    5. Compute a Theil-Sen line through the edges as a robust approximation of the edge
    6. Use a genetic algorithm to "stitch" the quadrants together

    Input:
    - Dictionary with parameters
    - Boolean value whether to plot intermediate results

    Output:
    - Final stitched image
    """

    # Make some directories for saving results
    current_res_name = get_resname(parameters["resolutions"][parameters["iteration"]])
    dirpath_tform = f"../results/" f"{parameters['patient_idx']}/" f"tform/"

    dirpath_images = f"../results/" f"{parameters['patient_idx']}/" f"images"

    dirpath_ga_progression = (
        f"../results/" f"{parameters['patient_idx']}/" f"ga_progression"
    )

    dirpath_ga_iteration = (
        f"../results/" f"{parameters['patient_idx']}/" f"ga_result_per_iteration"
    )

    dirpath_quadrants = (
        f"../results/"
        f"{parameters['patient_idx']}/"
        f"{parameters['slice_idx']}/"
        f"{current_res_name}/"
        f"quadrant"
    )

    for path in [
        dirpath_tform,
        dirpath_images,
        dirpath_ga_progression,
        dirpath_ga_iteration,
    ]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Check if optimized tform already exists
    parameters["filepath_tform"] = f"{dirpath_tform}/{current_res_name}_tform_final.npy"
    file_exists = os.path.isfile(parameters["filepath_tform"])

    # Start optimizing stitch
    if not file_exists:

        start_time = time.time()

        # Load previously saved quadrants
        with open(f"{dirpath_quadrants}_UL", "rb") as loadfile:
            quadrant_a = pickle.load(loadfile)
        with open(f"{dirpath_quadrants}_UR", "rb") as loadfile:
            quadrant_b = pickle.load(loadfile)
        with open(f"{dirpath_quadrants}_LL", "rb") as loadfile:
            quadrant_c = pickle.load(loadfile)
        with open(f"{dirpath_quadrants}_LR", "rb") as loadfile:
            quadrant_d = pickle.load(loadfile)

        # Save quadrants in list for easier handling of class methods
        quadrants = [quadrant_a, quadrant_b, quadrant_c, quadrant_d]

        # Load images
        print("- loading images...")
        for q in quadrants:
            q.load_images()

        # Get center of the quadrant
        for q in quadrants:
            q.get_image_center()

        # Perform initialization at the lowest resolution
        if parameters["iteration"] == 0:

            # Get bounding box based on the tissue mask
            for q in quadrants:
                q.get_bbox_corners(image=q.mask)

            # Get the initial transform consisting of rotation and cropping
            for q in quadrants:
                q.get_initial_transform()

            # Plot the rotation result as a visual check
            plot_rotation_result(
                quadrant_a=quadrant_a,
                quadrant_b=quadrant_b,
                quadrant_c=quadrant_c,
                quadrant_d=quadrant_d,
            )

            # Compute local transformation to align horizontal pieces. Input for this
            # method is the horizontal neighbour of this quadrant
            complementary_quadrants = [quadrant_b, quadrant_a, quadrant_d, quadrant_c]
            for q1, q2 in zip(quadrants, complementary_quadrants):
                q1.get_tformed_images_local(quadrant=q2)

            # Compute global transformation to align all pieces. Input for this method
            # is one of the vertical neighbours of this quadrant
            complementary_quadrants = [quadrant_c, quadrant_c, quadrant_a, quadrant_a]
            for q1, q2 in zip(quadrants, complementary_quadrants):
                q1.get_tformed_images_global(quadrant=q2)

            # Get final tform params for plotting later on
            initial_tform = dict()
            for q in quadrants:
                total_x = q.crop_trans_x + q.pad_trans_x + q.trans_x
                total_y = q.crop_trans_y + q.pad_trans_y + q.trans_y
                initial_tform[q.quadrant_name] = [
                    total_x,
                    total_y,
                    q.angle,
                    q.image_center_pre,
                    q.outshape,
                ]

            np.save(
                f"{dirpath_tform}/{current_res_name}_tform_initial.npy", initial_tform
            )

        # If initial transformation already exists, load and upsample it.
        elif parameters["iteration"] > 0:
            initial_tform = map_tform_low_res(parameters)

        # Apply transformation to the original images
        for q in quadrants:
            q.get_tformed_images(tform=initial_tform[q.quadrant_name])

        parameters["image_centers"] = [
            q.image_center_peri for q in quadrants
        ]  # required in cost function later on

        # Plot transformation result
        plot_transformation_result(
            quadrant_a=quadrant_a,
            quadrant_b=quadrant_b,
            quadrant_c=quadrant_c,
            quadrant_d=quadrant_d,
            parameters=parameters,
        )

        # Get edges from quadrants
        print(f"- extracting edges from images...")
        for q in quadrants:
            q.get_edges()

        # Compute Theil Sen lines through edges
        print("- computing Theil-Sen estimation of edge...")
        for q in quadrants:
            q.fit_theilsen_lines()

        # Plot all acquired Theil-Sen lines
        plot_theilsen_result(
            quadrant_a=quadrant_a,
            quadrant_b=quadrant_b,
            quadrant_c=quadrant_c,
            quadrant_d=quadrant_d,
            parameters=parameters,
        )

        # Optimization with genetic algorithm
        print("- computing reconstruction with genetic algorithm...")
        parameters["output_shape"] = quadrant_a.tform_image.shape

        ga_dict = genetic_algorithm(
            quadrant_a=quadrant_a,
            quadrant_b=quadrant_b,
            quadrant_c=quadrant_c,
            quadrant_d=quadrant_d,
            parameters=parameters,
            initial_tform=initial_tform,
        )
        np.save(parameters["filepath_tform"], ga_dict)

        # Plot the results of the genetic algorithm
        plot_ga_result(
            quadrant_a=quadrant_a,
            quadrant_b=quadrant_b,
            quadrant_c=quadrant_c,
            quadrant_d=quadrant_d,
            parameters=parameters,
            final_tform=ga_dict,
        )

        # Provide verbose on computation time
        end_time = time.time()
        current_res = parameters["resolutions"][parameters["iteration"]]
        sec = np.round(end_time - start_time, 1)
        print(
            f"> time to optimize patient {parameters['patient_idx']} at "
            f"resolution {current_res}: {sec} seconds"
        )

    else:
        print("- already optimized this resolution!")

    # At final resolution provide some extra visualizations
    if parameters["iteration"] == 3:

        # Make a gif of the tform result
        make_tform_gif(parameters)

        # Plot the fitness trajectory over the multiple resolutions
        plot_ga_multires(parameters)

    return
