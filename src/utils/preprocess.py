import os
import logging


def preprocess(fragments, parameters, log):
    """
    Function to load and preprocess all the tissue fragment images. The preprocessing
    mainly consists of resizing and saving the image at multiple resolutions.

    Input:
    - List of all fragments
    - Dict with parameters
    - Logging object for logging

    Output:
    - Fragment class with all loaded images
    """

    res_dir = f"{parameters['results_dir']}/images/{parameters['slice_idx']}/{parameters['res_name']}"
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    # Verify whether preprocessed fragments are available
    filepath = (
        f"{parameters['results_dir']}/images/"
        f"{parameters['slice_idx']}/"
        f"{parameters['res_name']}/"
        f"fragment_{fragments[-1].fragment_name}"
    )
    file_exists = os.path.isfile(filepath)

    # Preprocess all images if they are not yet available
    if not file_exists:

        for f in fragments:

            # Read fragment transformations
            f.read_transforms()

            # Read all original images
            f.read_image()

            # Preprocess (resize+pad) gray images
            f.preprocess_gray_image()

            # Preprocess (resize+pad) colour images
            f.preprocess_colour_image()

            # Segment tissue. This basically loads in the stored segmentations
            f.segment_tissue()

            # Apply mask to both gray and colour image
            f.apply_masks()

            # Save the fragment class for later use
            f.save_fragment()

        log.log(
            parameters["my_level"],
            f" - preprocessing resolution {parameters['resolutions'][parameters['iteration']]}"
        )

    # Else nothing, images will be loaded in the next step in the optimize stitch function
    else:
        log.log(
            parameters["my_level"],
            f" - already preprocessed resolution {parameters['resolutions'][parameters['iteration']]}"
        )

    return
