from .pairwise_alignment import run_pairwise_alignment
from .jigsawnet import jigsawnet_scoring
from .global_assembly import global_assembly
from .fetch_solutions import fetch_solutions
from .pairwise_alignment_utils import Fragment


def detect_configuration(parameters):
    """
    Function to find configuration of the different fragments.
    """

    parameters["log"].log(parameters["my_level"], "Detecting fragment configuration...")

    # Currently we support 2 or 4 fragments.
    if parameters["n_fragments"] == 2:

        # Retrieve the three best solutions in rank of likelihood
        solutions = get_configuration_2_fragments(parameters)

    elif parameters["n_fragments"] == 4:

        # Run initial pairwise alignment required for JigsawNet
        stitch_edge_file = parameters["save_dir"].joinpath(
            "configuration_detection",
            "stitch_edges.txt"
        )
        if not stitch_edge_file.exists():
            run_pairwise_alignment(parameters)

        # Score the fit of each fragment pair with Jigsawnet
        alignments_file = parameters["save_dir"].joinpath(
            "configuration_detection",
            "filtered_alignments.txt"
        )
        if not alignments_file.exists():
            jigsawnet_scoring(parameters)

        # Use the JigsawNet scores to determine feasible assemblies
        solution_file = parameters["save_dir"].joinpath(
            "configuration_detection",
            "location_solution.txt"
        )
        if not solution_file.exists():
            global_assembly(parameters)

        # Retrieve the k best solutions in rank of likelihood
        solutions = fetch_solutions(parameters)

    else:
        raise ValueError(
            f"Sorry, stitching {parameters['n_fragments']} fragments is not (yet) "
            f"supported. PythoStitcher currently only supports 2 or 4 fragments."
        )

    return solutions


def get_configuration_2_fragments(parameters):
    """
    Retrieve the configuration for 2 fragments. This is much easier than for 4 fragments,
    since we don't have to iterate over all potential combinations. There is only 1
    way in which these fragments can be fit together.
    """

    # Fetch all fragments
    fragment_names = sorted(
        [i.name for i in parameters["save_dir"].joinpath("preprocessed_images").iterdir()]
    )
    fragments = []
    for fragment in fragment_names:
        parameters["fragment_name"] = fragment
        fragments.append(Fragment(kwargs=parameters))

    # Preprocess all images
    parameters["log"].log(parameters["my_level"], " - identifying stitch edges")
    for f in fragments:
        f.read_images()
        f.process_images()
        f.save_images()
        f.get_stitch_edges()
        f.save_orientation()
        if f.require_landmark_computation:
            f.save_landmark_points()

    if fragments[0].force_config:
        print("Enforcing user-specified configuration")
        parameters["log"].log(parameters["my_level"], "Enforcing user-specified configuration")

    parameters["rot_steps"] = dict()
    rot_steps = [f.rot_k for f in fragments]
    for image, rot in zip(parameters["raw_image_names"], rot_steps):
        parameters["rot_steps"][str(image)] = rot

    location_solution_file = parameters["save_dir"].joinpath(
        "configuration_detection", "location_solution.txt"
    )
    with open(location_solution_file, "r") as f:
        # Read content and process
        contents = f.readlines()
        locations = [i.split(":")[-1].rstrip("\n") for i in contents]

    # Save original filenames and their location
    sol_dict = dict()
    original_filenames = parameters["raw_image_names"]

    for file, loc in zip(original_filenames, locations):
        sol_dict[file] = loc

    parameters["log"].log(parameters["my_level"], " - finished!\n")

    return [sol_dict]
