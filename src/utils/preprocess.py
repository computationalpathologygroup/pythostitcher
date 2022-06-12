import os


def preprocess(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters):
    """
    Function to load and preprocess all the quadrant images. The preprocessing mainly consists of resizing and saving
    the image at multiple resolutions.

    Input:
    - Class of all four quadrants
    - Dict with parameters

    Output:
    - Quadrant class with all loaded images

    """

    # Make directories for later saving
    dirnames = [f"../results",
                f"../results/{quadrant_D.patient_idx}",
                f"../results/{quadrant_D.patient_idx}/{quadrant_D.slice_idx}",
                f"../results/{quadrant_D.patient_idx}/{quadrant_D.slice_idx}/{quadrant_D.res_name}"]

    for name in dirnames:
        if not os.path.isdir(name):
            os.mkdir(name)

    # Verify whether preprocessed quadrants are available
    filepath = f"../results/" \
               f"{quadrant_D.patient_idx}/" \
               f"{quadrant_D.slice_idx}/" \
               f"{quadrant_D.res_name}/" \
               f"quadrant_{quadrant_D.quadrant_name}"
    file_exists = os.path.isfile(filepath)

    # Preprocess all images if they are not yet available
    if not file_exists:

        # Loop over all quadrants
        quadrants = [quadrant_A, quadrant_B, quadrant_C, quadrant_D]

        for q in quadrants:

            # Read all original images
            q.read_image()

            # Preprocess (resize+pad) gray images
            q.preprocess_gray_image()

            # Preprocess (resize+pad) colour images
            q.preprocess_colour_image()

            # Segment tissue. This basically loads in the stored segmentations
            q.segment_tissue()

            # Apply mask to both gray and colour image
            q.apply_masks()

            # Save the quadrant class for later use
            q.save_quadrant()

        print(f"- preprocessing resolution {parameters['resolutions'][parameters['iteration']]}")

    # Else nothing, images will be loaded in the next step in the optimize stitch function
    else:
        pass

    return
