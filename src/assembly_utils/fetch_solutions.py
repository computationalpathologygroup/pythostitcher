import numpy as np


def fetch_solutions(parameters):
    """
    Function to fetch the solution of the fragment configuration for cases with 4 fragments.
    """

    location_solution_file = parameters["save_dir"].joinpath("configuration_detection", "location_solution.txt")
    assert location_solution_file.exists(), "Could not find location_solution.txt file, " \
                                            "this is most likely because pythostitcher did not " \
                                            "find any feasible solutions. Check the " \
                                            "/configuration_detection directory to verify that " \
                                            "the fragment classification is correct."

    with open(location_solution_file, "r") as f:
        # Read content and process
        contents = f.readlines()
        contents = [i.rstrip("\n") for i in contents]
        contents = [i.split(",") for i in contents]

    # Sort mean squared error scores. Lower = better.
    scores = [int(i[0].split(":")[-1]) for i in contents]
    sort_idx = np.argsort(scores)

    # Gather all solutions
    final_solutions = []
    original_filenames = parameters["raw_image_names"]

    # Only save top k best results scored by mse for actual full-res stitching. You
    # can change this number based on your requirements.
    top_k = parameters["top_k"]
    for s in sort_idx[:top_k]:
        temp_sol = dict()

        # Relate each fragment pseudonym back to its original name
        for element in contents[s][1:]:

            # Sanity check to ensure good scores
            assert int(contents[s][0].lstrip("mse:")) == scores[s]

            idx = int(element.split(".png")[0].lstrip("fragment")) - 1
            filename = original_filenames[idx]
            location = element.split(":")[-1]
            temp_sol[filename] = location

        final_solutions.append(temp_sol)

    parameters["log"].log(
        parameters["my_level"], f"Examining {len(final_solutions)} best solutions...\n"
    )

    return final_solutions
