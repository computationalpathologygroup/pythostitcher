def preprocess(fragments, parameters):
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

    # Create new directories to save results where necessary
    new_dirs = [
        parameters["sol_save_dir"],
        parameters["sol_save_dir"].joinpath("highres", "blend_summary"),
        parameters["sol_save_dir"].joinpath("fragments"),
        parameters["sol_save_dir"].joinpath("tform"),
        parameters["sol_save_dir"].joinpath("images", "debug"),
        parameters["sol_save_dir"].joinpath("images", "ga_progression"),
        parameters["sol_save_dir"].joinpath("images", "ga_result_per_iteration"),
        parameters["sol_save_dir"].joinpath(
            "images", parameters["slice_idx"], parameters["res_name"]
        ),
    ]
    for d in new_dirs:
        if not d.is_dir():
            d.mkdir(parents=True)

    for f in fragments:

        # Read fragment transformations
        if parameters["n_fragments"] == 4:
            f.read_transforms()

        # Read all original images
        f.read_image()

        # Normalize the stain 
        f.normalize_stain()

        # Preprocess (resize+pad) gray images
        f.downsample_image()

        # Segment tissue. This basically loads in the stored segmentations
        f.segment_tissue()

        # Apply mask to both gray and colour image
        f.apply_masks()

        # Save the fragment class for later use
        f.save_fragment()

    # Save rot90 steps for later in high resolution reconstruction
    if not parameters["n_fragments"] == 2:
        parameters["rot_steps"] = dict()
        for f in fragments:
            parameters["rot_steps"][f.original_name] = f.rot_k

    return
