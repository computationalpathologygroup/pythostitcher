import glob
import re

from pythostitcher import run_pythostitcher


def setup_pythostitcher():
    """
    Main function to run PythoStitcher. Currently input parameters have to be specified manually in this file,
    but this could be replaced by argparse in a later version.
    """

    #### PATIENT NAME --- CHANGE ####
    patient_idx = 413
    patient_idx = "P" + str(patient_idx).zfill(6)

    # SPECIFY DIRECTORIES WHERE INPUT CAN BE FOUND
    data_dir = f"../radboud_data/{patient_idx}/images"
    results_dir = "../results"

    # Automatically get all quadrant filenames
    all_files = glob.glob(f"{data_dir}/*")
    filename_ul, filename_ur, filename_ll, filename_lr = None, None, None, None
    for file in all_files:

        # Use regex to get filename
        matched = re.search("DeepPCA", file)
        start = matched.span()[0]

        if "ul" in file:
            filename_ul = file[start:]
            filename_ul = filename_ul.split(".")[0]
        elif "ur" in file:
            filename_ur = file[start:]
            filename_ur = filename_ur.split(".")[0]
        elif "ll" in file:
            filename_ll = file[start:]
            filename_ll = filename_ll.split(".")[0]
        elif "lr" in file:
            filename_lr = file[start:]
            filename_lr = filename_lr.split(".")[0]

    filename_dict = dict()
    filename_dict["UL"] = filename_ul
    filename_dict["UR"] = filename_ur
    filename_dict["LL"] = filename_ll
    filename_dict["LR"] = filename_lr

    if None in filename_dict.values():
        raise ValueError("Error, could not find all quadrants")

    # OPTIONAL VARIABLES
    flip_quadrant = [0, 0, 0, 0]

    # Pack up all variables in a dictionary
    parameters = dict()
    parameters["filenames"] = filename_dict
    parameters["flipquadrants"] = flip_quadrant             # currently not in use
    parameters["data_dir"] = data_dir
    parameters["results_dir"] = results_dir
    parameters["patient_idx"] = patient_idx
    parameters["slice_idx"] = "test"                        # only relevant with multiple slices per patient

    # General input parameters
    parameters["resolutions"] = [0.05, 0.10, 0.25, 1.0]     # very important, can be tweaked
    parameters["pad_fraction"] = 0.7                        # required for not cutting corners

    # Parameters related to the cost function
    parameters["overunderlap_weights"] = [0.01, 0.01, 0.011, 0.01]
    parameters["cost_range"] = [0, 2]                                       # currently not in use
    parameters["overhang_penalty"] = parameters["cost_range"][-1]           # currently not in use
    parameters["cost_functions"] = ["simple_hists", "simple_hists",
                                    "simple_hists", "simple_hists"]
    parameters["cost_function_scaling"] = [res/parameters["resolutions"][0] for res in parameters["resolutions"]]
    parameters["nbins"] = 16
    parameters["hist_sizes"] = [4, 8, 20, 80]
    parameters["fraction_edge_length"] = 0.5
    parameters["outer_point_weight"] = 0.5
    parameters["overlap_weight"] = 100

    # Optimization parameters
    parameters["translation_ranges"] = [200*r for r in parameters["resolutions"]]
    parameters["angle_range"] = [10, 10, 10, 5]
    #parameters["sampling_deltas"] = [1, 1, 5, 10]
    parameters["sampling_deltas"] = [1, 2, 5, 10]
    parameters["GA_fitness"] = []

    # Start pythostitcher program
    run_pythostitcher(parameters)

    return


if __name__ == "__main__":
    setup_pythostitcher()
