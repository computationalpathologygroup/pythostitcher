import glob
import os
import argparse
import logging
import json

from utils.preprocess import preprocess
from utils.optimize_stitch import optimize_stitch
from utils.fragment_class import Fragment
from utils.high_resolution_writing import write_highres_fragments
from utils.high_resolution_blending import blend_image_tilewise
from utils.high_resolution_reconstruction import reconstruct_image
from utils.get_resname import get_resname
from utils.full_resolution import generate_full_res


def run_pythostitcher():
    """
    PythoStitcher is an automated and robust program for stitching prostate tissue
    fragments into a whole histological section.

    Original paper: https://www.nature.com/articles/srep29906
    Original Matlab code by Greg Penzias, 2016
    Python implementation by Daan Schouten, 2022

    Please see the 'sample data' directory for how to structure the input images for
    Pythostitcher. The general structure is as follows, where Pxxxxxx denotes the patient
    ID. The current version requires either two or four fragments, support for other
    amounts of fragments might be added in a future version.

    ___________________________
    /sample_data
        /Pxxxxxx
            /images [preprocessed]
                {fragment_name}.tif
                {fragment_name}.tif
            /masks [preprocessed]
                {fragment_name}.tif
                {fragment_name}.tif
            /raw_images
                {fragment_name}.mrxs
                {fragment_name}.mrxs
            /raw_masks
                {fragment_name}.tif
                {fragment_name}.tif
            rotations.txt
    ___________________________

    Input arguments
    * dictionary with parameters
    * image of each fragment (.tif)
    * tissue mask of each fragment (.tif)

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
    parser.add_argument(
        "--tissue", required=True, help="Anatomical region of tissue"
    )

    args = parser.parse_args()
    
    # Set variables
    patient_idx = args.patient
    tissue = args.tissue

    patient_idx = "P" + str(patient_idx).zfill(6)

    # Set data directories
    if any([patient_idx in i for i in glob.glob(f"../sample_data/{tissue}/*")]):
        data_dir = f"../sample_data/{tissue}/{patient_idx}"
        results_dir = f"../results/{tissue}/{patient_idx}"
    else:
        raise ValueError(
            f"Patient idx does not exist in location (../sample_data/{tissue})"
        )

    # Get valid filenames. Current list supports 2 or 4 pieces.
    with open("../config/config.json") as f:
        file = json.load(f)
        valid_fragments = file["valid_fragments"]

    # Get all fragment filenames
    all_files = [os.path.basename(i) for i in glob.glob(f"{data_dir}/images/*")]
    fragment_names = [i.split(".")[0].lower() for i in all_files]
    assert all(
        [i in valid_fragments for i in fragment_names]
    ), f"tissue fragment must be any of  {valid_fragments}, instead received {fragment_names}"

    # Pack up all variables in a dictionary
    parameters = dict()
    parameters["data_dir"] = data_dir
    parameters["results_dir"] = results_dir
    parameters["patient_idx"] = patient_idx
    parameters["slice_idx"] = "input"
    parameters["tissue"] = tissue
    parameters["fragment_names"] = fragment_names
    parameters["n_fragments"] = len(fragment_names)
    assert parameters["n_fragments"] in [
        2, 4
    ], "only stitching of 2 or 4 pieces is currently supported in pythostitcher"
    parameters["my_level"] = 35

    # General input parameters. The resolutions can be changed depending on the size of th
    # e
    # input images. The two most important aspects are that a) the resolutions should be in
    # ascending order and b) that the first resolution should be of sufficient size to allow
    # for a reasonable initialization of the stitching.

    parameters["resolutions"] = [0.025, 0.05, 0.15, 0.5]
    if parameters["tissue"] == "oesophagus":
        parameters["resolutions"] = [i/2 for i in parameters["resolutions"]]
    parameters["image_level"] = 6 if parameters["tissue"] == "prostate" else 4
    parameters["pad_fraction"] = 0.5


    # Genetic algorithm parameters. Check the genetic_algorithm.py file for the exact
    # implementation of these parameters and check the pygad documentation
    # (https://pygad.readthedocs.io/en/latest/) for an explanation and other options
    # for all parameters.
    parameters["n_solutions"] = 20
    parameters["n_generations"] = [200, 150, 100, 100]
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

    # Make directories for later saving
    dirnames = [
        f"../results/{tissue}",
        f"{parameters['results_dir']}",
        f"{parameters['results_dir']}/highres",
        f"{parameters['results_dir']}/highres/blend_summary",
        f"{parameters['results_dir']}/tform",
        f"{parameters['results_dir']}/fragments",
        f"{parameters['results_dir']}/images",
        f"{parameters['results_dir']}/images/debug",
        f"{parameters['results_dir']}/images/ga_progression",
        f"{parameters['results_dir']}/images/ga_result_per_iteration",
        f"{parameters['results_dir']}/images/{parameters['slice_idx']}",
    ]

    for name in dirnames:
        if not os.path.isdir(name):
            os.mkdir(name)

    # Remove previous logfile if applicable
    logfile = f"{results_dir}/pythostitcher_log.log"
    if os.path.isfile(logfile):
        os.remove(logfile)

    # Initiate logging file
    logging.basicConfig(
        filename=logfile,
        level=logging.WARNING,
        format="%(asctime)s    %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.addLevelName(parameters["my_level"], "output")
    log = logging.getLogger("pythostitcher")
    log.log(parameters["my_level"], "Preprocessing data on multiple resolutions...")

    # Start with preprocessing data
    print("\nComputing Pythostitcher transformation")

    for c, res in enumerate(parameters["resolutions"]):

        # Set current iteration
        parameters["iteration"] = c
        parameters["res_name"] = get_resname(res)

        fragments = []
        for fragment in fragment_names:
            fragments.append(Fragment(fragment_name=fragment, kwargs=parameters))

        # Preprocess all images
        preprocess(fragments=fragments, parameters=parameters, log=log)

    log.log(parameters["my_level"], " > finished!\n")

    # Optimize stitch for multiple resolutions
    for i in range(len(parameters["resolutions"])):

        # Set current iteration
        parameters["iteration"] = i
        log.log(parameters["my_level"], f"Optimizing stitch at resolution {parameters['resolutions'][i]}")
        optimize_stitch(parameters=parameters, log=log)

    # Generate full resolution blended image
    generate_full_res(parameters=parameters, log=log)

    log.log(
        parameters["my_level"],
        f"Succesfully stitched {parameters['tissue']} {parameters['patient_idx']}"
    )

    return


if __name__ == "__main__":
    run_pythostitcher()
