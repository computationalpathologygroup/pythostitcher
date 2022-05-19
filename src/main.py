from pythostitcher import setup_pythostitcher
import glob


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
        if "ul" in file:
            filename_ul = file.split("/")[-1]
        elif "ur" in file:
            filename_ur = file.split("/")[-1]
        elif "ll" in file:
            filename_ll = file.split("/")[-1]
        elif "lr" in file:
            filename_lr = file.split("/")[-1]

    filename_dict = dict()
    filename_dict["UL"] = filename_ul
    filename_dict["UR"] = filename_ur
    filename_dict["LL"] = filename_ll
    filename_dict["LR"] = filename_lr

    if None in filename_dict.values():
        raise ValueError("Error, could not find all quadrants")

    # OPTIONAL VARIABLES
    flip_quadrant = [0, 0, 0, 0]

    # Pack up all variables in a dictionary to pass on
    parameters = dict()
    parameters["filenames"] = filename_dict
    parameters["flipquadrants"] = flip_quadrant
    parameters["data_dir"] = data_dir
    parameters["results_dir"] = results_dir
    parameters["patient_idx"] = patient_idx
    parameters["slice_idx"] = "slice"

    # Start pythostitcher program
    setup_pythostitcher(parameters)

    return


if __name__ == "__main__":
    run_pythostitcher()
