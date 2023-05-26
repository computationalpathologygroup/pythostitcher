import argparse
import json
import logging
import os
import pathlib

from assembly_utils.detect_configuration import detect_configuration
from preprocessing_utils.prepare_data import prepare_data
from pythostitcher_utils.fragment_class import Fragment
from pythostitcher_utils.full_resolution import generate_full_res
from pythostitcher_utils.get_resname import get_resname
from pythostitcher_utils.optimize_stitch import optimize_stitch
from pythostitcher_utils.preprocess import preprocess

os.environ["VIPS_CONCURRENCY"] = "20"


def load_parameter_configuration(data_dir, save_dir, output_res):
    """
    Convenience function to load all the PythoStitcher parameters and pack them up
    in a dictionary for later use.
    """

    # Verify its existence
    config_file = pathlib.Path().absolute().parent.joinpath("config/parameter_config.json")
    assert config_file.exists(), "parameter config file not found"

    # Load main parameter config
    with open(config_file) as f:
        parameters = json.load(f)

    # Convert model weight paths to absolute paths
    parameters["weights_fragment_classifier"] = (
        pathlib.Path().absolute().parent.joinpath(parameters["weights_fragment_classifier"])
    )
    parameters["weights_jigsawnet"] = (
        pathlib.Path().absolute().parent.joinpath(parameters["weights_jigsawnet"])
    )

    # Insert parsed arguments
    parameters["data_dir"] = data_dir
    parameters["save_dir"] = save_dir
    parameters["patient_idx"] = data_dir.name
    parameters["output_res"] = output_res
    parameters["fragment_names"] = sorted([
        i.name for i in data_dir.joinpath("raw_images").iterdir() if not i.is_dir()
    ])
    parameters["n_fragments"] = len(parameters["fragment_names"])
    parameters["resolution_scaling"] = [
        i / parameters["resolutions"][0] for i in parameters["resolutions"]
    ]

    parameters["raw_image_names"] = sorted(
        [i.name for i in data_dir.joinpath("raw_images").iterdir() if not i.is_dir()]
    )
    if data_dir.joinpath("raw_masks").is_dir():
        parameters["raw_mask_names"] = sorted(
            [i.name for i in data_dir.joinpath("raw_masks").iterdir()]
        )
    else:
        parameters["raw_mask_names"] = [None] * len(parameters["raw_image_names"])

    # Some assertions
    assert parameters["n_fragments"] in [
        2, 4,
    ], "pythostitcher only supports stitching 2/4 fragments"

    # Make directories for later saving
    dirnames = [
        pathlib.Path(parameters["save_dir"]),
        pathlib.Path(parameters["save_dir"]).joinpath("configuration_detection", "checks"),
    ]

    for d in dirnames:
        if not d.is_dir():
            d.mkdir(parents=True)

    return parameters


def collect_arguments():
    """
    Function to parse arguments into main function
    """

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Stitch histopathology images into a pseudo whole-mount image"
    )
    parser.add_argument(
        "--datadir", required=True, type=pathlib.Path, help="Path to the case to stitch"
    )
    parser.add_argument(
        "--savedir", required=True, type=pathlib.Path, help="Directory to save the results",
    )
    parser.add_argument(
        "--resolution",
        required=True,
        default=0.25,
        type=float,
        help="Output resolution (µm/pixel) of the reconstructed image. Should be roughly "
        "in range of 0.25-20.",
    )
    args = parser.parse_args()

    # Extract arguments
    data_dir = pathlib.Path(args.datadir)
    save_dir = pathlib.Path(args.savedir).joinpath(data_dir.name)
    resolution = args.resolution

    assert data_dir.is_dir(), "provided patient directory doesn't exist"
    assert data_dir.joinpath("raw_images").is_dir(), "patient has no 'raw_images' directory"
    assert (
        len(list(data_dir.joinpath("raw_images").iterdir())) > 0
    ), "no images found in 'raw_images' directory"
    assert resolution > 0, "output resolution cannot be negative"

    print(
        f"\nRunning job with following parameters:"
        f"\n - Data dir: {data_dir}"
        f"\n - Save dir: {save_dir}"
        f"\n - Output resolution: {resolution} µm/pixel\n"
    )

    return data_dir, save_dir, resolution


def main():
    """
    PythoStitcher is an automated and robust program for stitching prostate tissue
    fragments into a whole histological section.

    Original paper: https://www.nature.com/articles/srep29906
    Original Matlab code by Greg Penzias, 2016
    Python implementation by Daan Schouten, 2022

    Please see the data directory for how to structure the input images for
    Pythostitcher. The general structure is as follows, where Patient_identifier denotes
    any (anonymized) patient identifier. The current version requires either two or four
    fragments, support for other amounts of fragments might be added in a future version.

    ___________________________
    /data
        /{Patient_identifier}
            /raw_images
                {fragment_name}.mrxs§
                {fragment_name}.mrxs
            /raw_masks
                {fragment_name}.tif
                {fragment_name}.tif
    ___________________________

    """

    ### ARGUMENT CONFIGURATION ###
    # Collect arguments
    data_dir, save_dir, output_res = collect_arguments()
    parameters = load_parameter_configuration(data_dir, save_dir, output_res)

    # Initiate logging file
    logfile = save_dir.joinpath("pythostitcher_log.log")
    if logfile.exists():
        logfile.unlink()

    logging.basicConfig(
        filename=logfile,
        level=logging.WARNING,
        format="%(asctime)s    %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.addLevelName(parameters["my_level"], "output")
    log = logging.getLogger("pythostitcher")

    parameters["log"] = log
    parameters["log"].log(parameters["my_level"], f"Running job with following parameters:")
    parameters["log"].log(parameters["my_level"], f" - Data dir: {parameters['data_dir']}")
    parameters["log"].log(parameters["my_level"], f" - Save dir: {parameters['save_dir']}")
    parameters["log"].log(
        parameters["my_level"], f" - Output resolution: {parameters['output_res']}\n"
    )

    if not data_dir.joinpath("raw_masks").is_dir():
        parameters["log"].log(
            parameters["my_level"],
            f"WARNING: PythoStitcher did not find any raw tissuemasks. If you intend to use "
            f"PythoStitcher with pregenerated tissuemasks, please put these files in "
            f"[{data_dir.joinpath('raw_masks')}]. If no tissuemasks are supplied, "
            f"PythoStitcher will use a generic tissue segmentation which may not perform "
            f"as well as the AI-generated masks. In addition, PythoStitcher will not "
            f"be able to generate the full resolution end result.",
        )

    ### MAIN PYTHOSTITCHER #s##
    # Preprocess data
    prepare_data(parameters=parameters)

    # Detect configuration of fragments. Return the 3 most likely configurations in order
    # of likelihood.
    solutions = detect_configuration(parameters=parameters)

    # Loop over all solutions
    for count_sol, sol in enumerate(solutions, 1):
        parameters["log"].log(parameters["my_level"], f"### Exploring solution {count_sol} ###")
        parameters["detected_configuration"] = sol
        parameters["num_sol"] = count_sol
        parameters["sol_save_dir"] = parameters["save_dir"].joinpath(f"sol_{count_sol}")

        for count_res, res in enumerate(parameters["resolutions"]):

            # Set current iteration
            parameters["iteration"] = count_res
            parameters["res_name"] = get_resname(res)
            parameters["fragment_names"] = [sol[i].lower() for i in sorted(sol)]

            fragments = []
            for im_path, fragment_name in sol.items():
                fragments.append(
                    Fragment(im_path=im_path, fragment_name=fragment_name, kwargs=parameters)
                )

            # Preprocess all images to a usable format for PythoStitcher
            preprocess(fragments=fragments, parameters=parameters)

            # Get optimal stitch using a genetic algorithm
            optimize_stitch(parameters=parameters)

        # Generate full resolution blended image
        generate_full_res(parameters=parameters, log=log)

        parameters["log"].log(
            parameters["my_level"], f"### Succesfully stitched solution {count_sol} ###\n",
        )

    return


if __name__ == "__main__":
    main()
