import time
import pickle
import cv2

from .get_resname import get_resname
from .genetic_algorithm import genetic_algorithm
from .map_tform_low_res import map_tform_low_res
from .plot_tools import *
from .adjust_final_rotation import adjust_final_rotation
from .transformations import warp_image
from .fuse_images_lowres import fuse_images_lowres


def optimize_stitch(parameters):
    """
    Function to optimize the stitching between fragments. This will consist of the
    following steps:
    1. Compute the smallest bounding box around the fragment
    2. Rotate the fragment as a first step towards alignment
    3. Globally align the fragment such that they share the same coordinate system
    4. Identify cornerpoints in the fragment and extract relevant edges
    5. Compute a Theil-Sen line through the edges as a robust approximation of the edge
    6. Use a genetic algorithm to "stitch" the fragments together

    Input:
    - Dictionary with parameters
    - Logging object

    Output:
    - Final stitched image
    """

    parameters["log"].log(parameters["my_level"], f"Optimizing stitch at resolution {parameters['resolutions'][parameters['iteration']]}")

    current_res_name = get_resname(parameters["resolutions"][parameters["iteration"]])
    dirpath_tform = f"{parameters['sol_save_dir']}/tform"

    dir_fragments = (
        f"{parameters['sol_save_dir']}/images/{parameters['slice_idx']}/"
        f"{current_res_name}"
    )

    # Check if optimized tform already exists
    parameters["filepath_tform"] = f"{dirpath_tform}/{current_res_name}_tform_final.npy"
    file_exists = os.path.isfile(parameters["filepath_tform"])

    # Start optimizing stitch
    if not file_exists:

        start_time = time.time()
        fragment_info = dict()

        for fragment in parameters["fragment_names"]:

            with open(f"{dir_fragments}/fragment_{fragment}", "rb") as f:
                fragment_info_file = pickle.load(f)
            fragment_info[str(fragment)] = fragment_info_file
            f.close()

        fragments = list(fragment_info.values())

        # Load images
        parameters["log"].log(parameters["my_level"], " - loading images...")
        for f in fragments:
            f.load_images()

        # Get center of the fragment
        for f in fragments:
            f.get_image_center()

        # Perform initialization at the lowest resolution
        if parameters["iteration"] == 0:

            # Get bounding box based on the tissue mask
            for f in fragments:
                f.get_bbox_corners(image=f.mask)

            # Get the initial transform consisting of rotation and cropping
            for f in fragments:
                f.get_initial_transform()

            # Plot the rotation result as a visual check
            plot_rotation_result(fragments=fragments, parameters=parameters)

            # Irrespective of number of fragments, we always need to compute an initial
            # pairwise transformation. In case of 2 fragments, this is also the final
            # initialization. In case of 4 fragments, we also need to compute an additional
            # pairwise transformation of the two formed pairs
            for f in fragments:
                f.get_tformed_images_pair(fragments=fragments)

            if parameters["n_fragments"] == 4:
                for f in fragments:
                    f.get_tformed_images_total(fragments=fragments)

            # Get final tform params for plotting later on
            initial_tform = dict()
            for f in fragments:
                total_x = f.crop_trans_x + f.pad_trans_x + f.trans_x
                total_y = f.crop_trans_y + f.pad_trans_y + f.trans_y
                initial_tform[f.final_orientation] = [
                    total_x,
                    total_y,
                    f.angle,
                    f.image_center_pre,
                    f.output_shape,
                ]

            np.save(
                f"{dirpath_tform}/{current_res_name}_tform_initial.npy", initial_tform
            )

        # If initial transformation already exists, load and upsample it.
        elif parameters["iteration"] > 0:
            initial_tform = map_tform_low_res(parameters)

        # Apply transformation to the original images
        for f in fragments:
            f.get_tformed_images(tform=initial_tform[f.final_orientation])

        # Required for cost function
        parameters["image_centers"] = [f.image_center_peri for f in fragments]

        # Plot transformation result
        plot_transformation_result(fragments=fragments, parameters=parameters)

        # Get edges from fragments
        parameters["log"].log(parameters["my_level"], f" - extracting edges from images...")
        for f in fragments:
            f.get_edges()

        # Compute Theil Sen lines through edges
        parameters["log"].log(parameters["my_level"], " - computing Theil-Sen estimation of edge...")
        for f in fragments:
            f.fit_theilsen_lines()

        # Plot all acquired Theil-Sen lines
        plot_theilsen_result(fragments=fragments, parameters=parameters)

        # Optimization with genetic algorithm
        parameters["log"].log(parameters["my_level"], " - computing reconstruction with genetic algorithm...")
        parameters["output_shape"] = fragments[0].tform_image.shape

        ga_dict = genetic_algorithm(
            fragments=fragments,
            parameters=parameters,
            initial_tform=initial_tform,
        )
        np.save(parameters["filepath_tform"], ga_dict)

        # Get final transformed image per fragment
        all_images = []
        for f in fragments:
            final_tform = ga_dict[f.final_orientation]
            f.final_image = warp_image(
                src=f.colour_image_original,
                center=final_tform[3],
                rotation=final_tform[2],
                translation=final_tform[:2],
                output_shape=final_tform[4],
            )
            all_images.append(f.final_image)

        # Get final fused image, correct for the rotation and display it
        final_image = fuse_images_lowres(images=all_images, parameters=parameters)
        # final_image = adjust_final_rotation(image=final_image)
        plot_ga_result(final_image=final_image, parameters=parameters)

        # Provide verbose on computation time
        end_time = time.time()
        current_res = parameters["resolutions"][parameters["iteration"]]
        sec = np.round(end_time - start_time, 1)
        parameters["log"].log(
            parameters["my_level"],
            f" > time to optimize "
            f"resolution {current_res}: {sec} seconds\n"
        )

        # At final resolution provide some extra visualizations
        if parameters["iteration"] == 3:

            # Save the final result
            r, c = np.nonzero((final_image[:, :, 0]>3)*1)
            cv2.imwrite(
                f"{parameters['sol_save_dir']}/images/GA_endresult.png",
                cv2.cvtColor(final_image[np.min(r):np.max(r), np.min(c):np.max(c), :], cv2.COLOR_RGB2BGR)
            )

            # Make a gif of the tform result
            make_tform_gif(parameters)

            # Plot the fitness trajectory over the multiple resolutions
            plot_ga_multires(parameters)

            # Save all fragments and their info
            # if not os.path.isfile(f"{parameters['sol_save_dir']}/fragments/parameters"):
            #     for f in fragments:
            #         with open(f"{parameters['sol_save_dir']}/fragments/{f.final_orientation}", "wb") as savefile:
            #             pickle.dump(f, savefile)
            #
            #     with open(f"{parameters['sol_save_dir']}/fragments/parameters", "wb") as savefile:
            #         pickle.dump(parameters, savefile)

    else:
        parameters["log"].log(parameters["my_level"], " - already optimized this resolution!\n")


    return
