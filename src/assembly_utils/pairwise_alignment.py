from .pairwise_alignment_utils import *
from .fragment_classifier import Classifier


def run_pairwise_alignment(parameters):
    """
    Pairwise alignment function to compute pairs of corresponding images which serve
    as the input for the JigsawNet model.
    """

    # Get all fragment filenames
    fragment_names = sorted(
        [i.name for i in parameters["save_dir"].joinpath("preprocessed_images").iterdir()]
    )
    assert len(fragment_names) > 0, "no fragments were found in the given directory"

    # Insert some more variables
    parameters["pa_fragment_names"] = fragment_names
    parameters["pa_resolution"] = [0.1]

    # Create fragment list .txt file
    with open(
        parameters["save_dir"].joinpath("configuration_detection", "fragment_list.txt"), "w"
    ) as f:
        for name in fragment_names:
            f.write(f"{name}\n")

    # Create background colour .txt file
    with open(parameters["save_dir"].joinpath("configuration_detection", "bg_color.txt"), "w") as f:
        f.write("0 0 0")

    # Get fragment classifier model
    classifier = Classifier(weights=parameters["weights_fragment_classifier"])
    classifier.build_model()
    parameters["fragment_classifier"] = classifier

    # Fetch all fragments
    fragments = []
    for fragment in fragment_names:
        parameters["fragment_name"] = fragment
        fragments.append(Fragment(kwargs=parameters))

    # Preprocess all images
    parameters["log"].log(parameters["my_level"], " - identifying stitch edges")
    for f in fragments:
        f.read_images()
        f.process_images()
        f.classify_stitch_edges()
        f.save_images()
        f.get_stitch_edges()
        if f.require_landmark_computation:
            f.save_landmark_points()

    plot_stitch_edge_classification(fragments=fragments, parameters=parameters)

    # Find matching pairs
    parameters["log"].log(parameters["my_level"], f" - computing pairwise alignment")
    explore_pairs(fragments=fragments, parameters=parameters)

    return
