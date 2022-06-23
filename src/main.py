import glob
import os
import argparse

from utils.preprocess import preprocess
from utils.optimize_stitch import optimize_stitch
from utils.quadrant_class import Quadrant


def run_pythostitcher():
    """
    PythoStitcher is an automated and robust program for stitching prostate tissue
    fragments into a whole histological section.

    Original paper: https://www.nature.com/articles/srep29906
    Original Matlab code by Greg Penzias, 2016
    Python implementation by Daan Schouten, 2022

    Please see the 'sample data' directory for how to structure the input images for
    Pythostitcher. The general structure is as follows, where Pxxxxxx denotes the patient
    ID. The current version requires four quadrants, support for less/more quadrants
    might be added in a future version.

    ___________________________
    /sample_data
        /Pxxxxxx
            /images
                ul.tif
                ur.tif
                ll.tif
                lr.tif
            /masks
                ul.tif
                ur.tif
                ll.tif
                lr.tif
    ___________________________

    Input arguments
    * dictionary with parameters
    * image of each quadrant (.tif)
    * tissue mask of each quadrant (.tif)

    Output
    * Final reconstructed image

    """

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Stitch prostate histopathology images into a pseudo whole-mount image"
    )
    parser.add_argument(
        "--patient", type=int, required=True, help="Index of the patient to analyse"
    )
    args = parser.parse_args()

    # Set variables
    patient_idx = args.patient                      # patient number between 1 and 999999
    patient_idx = "P" + str(patient_idx).zfill(6)   # convert int to string for later saving

    # Set data directories
    if any([patient_idx in i for i in glob.glob("../sample_data/*")]):
        data_dir = f"../sample_data/{patient_idx}/images"  # data directory with images
        results_dir = "../results"                         # directory to save results
    else:
        raise ValueError("Patient idx does not exist in location (../sample_data/)")

    # Get all quadrant filenames
    all_files = [os.path.basename(i) for i in glob.glob(f"{data_dir}/*")]
    filename_ul, filename_ur, filename_ll, filename_lr = None, None, None, None

    for file in all_files:
        if "ul" in file:
            filename_ul = file
            filename_ul = filename_ul.split(".")[0]
        elif "ur" in file:
            filename_ur = file
            filename_ur = filename_ur.split(".")[0]
        elif "ll" in file:
            filename_ll = file
            filename_ll = filename_ll.split(".")[0]
        elif "lr" in file:
            filename_lr = file
            filename_lr = filename_lr.split(".")[0]

    filename_dict = dict()
    keys = ["UL", "UR", "LL", "LR"]
    filenames = [filename_ul, filename_ur, filename_ll, filename_lr]
    for key, filename in zip(keys, filenames):
        filename_dict[key] = filename

    if None in filename_dict.values():
        raise ValueError("Error, could not find all quadrants")

    # Pack up all variables in a dictionary
    parameters = dict()
    parameters["filenames"] = filename_dict
    parameters["data_dir"] = data_dir
    parameters["results_dir"] = results_dir
    parameters["patient_idx"] = patient_idx
    parameters["slice_idx"] = "preprocessed_input"

    # General input parameters. The resolutions can be changed depending on the size of the
    # input images. The two most important aspects are that a) the resolutions should be in
    # ascending order and b) that the first resolution should be of sufficient size to allow
    # for a reasonable initialization of the stitching.
    parameters["resolutions"] = [0.025, 0.05, 0.15, 0.5]
    parameters["pad_fraction"] = 0.7

    # Genetic algorithm parameters. Check the genetic_algorithm.py file for the exact
    # implementation of these parameters and check the pygad documentation
    # (https://pygad.readthedocs.io/en/latest/) for an explanation and other options
    # for all parameters.
    parameters["n_solutions"] = 20
    parameters["n_generations"] = 100
    parameters["n_parents"] = 3
    parameters["n_mating"] = 6
    parameters["p_crossover"] = 0.5
    parameters["crossover_type"] = "scattered"
    parameters["p_mutation"] = 0.25
    parameters["mutation_type"] = "random"
    parameters["parent_selection"] = "rank"

    # Parameters related to the cost function
    parameters["resolution_scaling"] = [
        res / parameters["resolutions"][0] for res in parameters["resolutions"]
    ]
    parameters["nbins"] = 16
    parameters["hist_sizes"] = [4, 8, 20, 80]
    parameters["outer_point_weight"] = 0.5
    parameters["overlap_weight"] = 100
    parameters["distance_scaling_hor_required"] = True
    parameters["distance_scaling_ver_required"] = True

    # Optimization parameters
    parameters["translation_range"] = [
        0.05,
        0.05 / (parameters["resolutions"][1] / parameters["resolutions"][0]),
        0.05 / (parameters["resolutions"][2] / parameters["resolutions"][0]),
        0.05 / (parameters["resolutions"][2] / parameters["resolutions"][0]),
    ]
    parameters["angle_range"] = [10, 10, 5, 5]
    parameters["GA_fitness"] = []

    # Start with preprocessing data
    print("\nPreprocessing data on multiple resolutions...")
    for i in range(len(parameters["resolutions"])):

        # Set current iteration
        parameters["iteration"] = i

        # Initiate all quadrants
        quadrant_a = Quadrant(quadrant_name="UL", kwargs=parameters)
        quadrant_b = Quadrant(quadrant_name="UR", kwargs=parameters)
        quadrant_c = Quadrant(quadrant_name="LL", kwargs=parameters)
        quadrant_d = Quadrant(quadrant_name="LR", kwargs=parameters)

        # Preprocess all images
        preprocess(
            quadrant_a=quadrant_a,
            quadrant_b=quadrant_b,
            quadrant_c=quadrant_c,
            quadrant_d=quadrant_d,
            parameters=parameters,
        )

    print("> Finished!")

    # Optimize stitch for multiple resolutions
    for i in range(len(parameters["resolutions"])):

        # Set current iteration
        parameters["iteration"] = i

        print(f"\nOptimizing stitch at resolution {parameters['resolutions'][i]}")
        optimize_stitch(parameters=parameters)

    return


if __name__ == "__main__":
    run_pythostitcher()
