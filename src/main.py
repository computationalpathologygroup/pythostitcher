import glob
import re

from pythostitcher import setup_pythostitcher



def run_pythostitcher():
    """
    Main function to run PythoStitcher. Currently input parameters have to be specified manually in this file,
    but this could be replaced by argparse in a later version.
    """

    #### PATIENT NAME --- CHANGE ####
    patient_idx = 413
    patient_idx = "P" + str(patient_idx).zfill(6)

    # SPECIFY DIRECTORIES WHERE INPUT CAN BE FOUND
    data_dir = (f"../radboud_data/{patient_idx}/images")
    results_dir = ("../results")

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
    parameters["flipquadrants"] = flip_quadrant
    parameters["data_dir"] = data_dir
    parameters["results_dir"] = results_dir
    parameters["patient_idx"] = patient_idx
    parameters["slice_idx"] = "test"

    # General input parameters
    parameters["debug"] = False
    parameters["ishpc"] = False
    parameters["display_opts"] = "EdgesAndLines"
    parameters["rev_name"] = "r1"
    parameters["resolutions"] = [0.05, 0.10, 0.25, 1.0]
    parameters["padsizes"] = [int(200*r) for r in parameters["resolutions"]]

    # Parameters related to the cost function
    parameters["overunderlap_weights"] = [0.01, 0.01, 0.011, 0.01]
    parameters["cost_range"] = [0, 2]
    parameters["overhang_penalty"] = parameters["cost_range"][-1]
    # parameters["cost_functions"] = ["raw_intensities", "raw_intensities", "simple_hists", "simple_hists"]
    parameters["cost_functions"] = ["simple_hists", "simple_hists", "simple_hists", "simple_hists"]
    parameters["nbins"] = 16
    # parameters["hist_sizes"] = [[1, 1], [1, 1], [21, 11], [81, 41]]
    parameters["hist_sizes"] = [[5, 3], [11, 5], [21, 11], [81, 41]]
    parameters["fraction_edge_length"] = 0.5
    parameters["outer_point_weight"] = 0.6

    # Optimization parameters
    parameters["translation_ranges"] = [1000*r for r in parameters["resolutions"]]
    #parameters["sampling_deltas"] = [1, 1, 5, 10]
    parameters["sampling_deltas"] = [1, 2, 5, 10]
    parameters["angle_range"] = 15

    # Start pythostitcher program
    setup_pythostitcher(parameters)

    return


if __name__ == "__main__":
    run_pythostitcher()
