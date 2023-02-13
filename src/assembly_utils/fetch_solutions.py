import numpy as np


def fetch_solutions(parameters):
    """
    Function to fetch the solution of the fragment configuration for cases with 4 fragments.
    """

    with open(parameters["save_dir"].joinpath("configuration_detection", "location_solution.txt"), "r") as f:
        # Read content and process
        contents = f.readlines()
        contents = [i.rstrip("\n") for i in contents]
        contents = [i.split(",") for i in contents]

        # Sort mean squared error scores. Lower = better.
        scores = [int(i[0].split(":")[-1]) for i in contents]
        sort_idx = np.argsort(scores)

        # Gather all solutions
        final_solutions = []
        original_filenames = sorted([i.name for i in parameters["data_dir"].joinpath("raw_images").iterdir() if not i.is_dir()])

        # Only save 3 best results
        for c, idx in enumerate(sort_idx[:3]):
            temp_sol = dict()

            # Relate each fragment pseudonym back to its original name
            for element in contents[c][1:]:
                idx = int(element.split(":")[0].split("_")[-1].rstrip(".png")) - 1
                filename = original_filenames[idx]
                location = element.split(":")[-1]
                temp_sol[filename] = location

            final_solutions.append(temp_sol)

    parameters["log"].log(parameters["my_level"], f"Examining {len(final_solutions)} best solutions...\n")

    return final_solutions
