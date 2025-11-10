import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from assembly_utils.detect_configuration import detect_configuration
from preprocessing_utils.prepare_data import prepare_data
from pythostitcher_utils.fragment_class import Fragment
from pythostitcher_utils.full_resolution import generate_full_res
from pythostitcher_utils.get_resname import get_resname
from pythostitcher_utils.optimize_stitch import optimize_stitch
from pythostitcher_utils.preprocess import preprocess


def _write_completion_marker(save_dir: Path, marker_name: str = ".complete"):
    """
    Atomically write a completion marker in the case save directory.
    """
    tmp_path = save_dir.joinpath(f"{marker_name}.tmp")
    final_path = save_dir.joinpath(marker_name)
    with open(tmp_path, "w") as f:
        f.write("ok\n")
    os.replace(tmp_path, final_path)


def _copy_file_to_local(source_path, local_dir):
    """
    Copy a file from NAS to local directory.
    
    Args:
        source_path: Path to source file on NAS
        local_dir: Local directory (e.g., /opt/input)
    
    Returns:
        Path to the local copy
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = local_dir.joinpath(source_path.name)
    
    try:
        shutil.copy2(source_path, local_path)
    except Exception as e:
        raise RuntimeError(f"Failed to copy {source_path} to {local_path}: {e}")
    
    return local_path


def _cleanup_local_directory(local_dir):
    """
    Remove local directory and all its contents.
    
    Args:
        local_dir: Directory to remove
    """
    local_dir = Path(local_dir)
    if local_dir.exists():
        try:
            shutil.rmtree(local_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to cleanup {local_dir}: {e}")


def _load_base_parameters(config_path):
    """
    Load JSON config and convert model weight paths to absolute.
    """
    
    config_file = Path(config_path)
    assert config_file.exists(), f"parameter config file not found at {config_file}"

    with open(config_file) as f:
        parameters = json.load(f)

    # Convert relative paths to absolute paths relative to config file location
    config_dir = config_file.parent
    
    parameters["weights_fragment_classifier"] = (
        config_dir.parent.joinpath(parameters["weights_fragment_classifier"])
    )
    parameters["weights_jigsawnet"] = (
        config_dir.parent.joinpath(parameters["weights_jigsawnet"])
    )
    
    return parameters


def _setup_logging(save_dir, my_level):
    """
    Initialize logging for a case.
    """
    
    logfile = save_dir.joinpath("pythostitcher_log.txt")
    if logfile.exists():
        logfile.unlink()

    logging.basicConfig(
        filename=logfile,
        level=logging.WARNING,
        format="%(asctime)s    %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    logging.addLevelName(my_level, "output")
    
    return logging.getLogger(f"{save_dir.name}")


def load_parameters(image_paths, mask_paths, save_path, force_config_path, landmark_paths, base_parameters):
    """
    Build parameters dict from dataframe-provided paths.
    """
    
    parameters = base_parameters.copy()

    parameters["save_dir"] = save_path
    parameters["patient_idx"] = save_path.name

    # Store original paths for reference
    parameters["original_image_paths"] = [Path(p) for p in image_paths]
    parameters["original_mask_paths"] = [Path(p) for p in mask_paths]
    
    # Handle read_local: copy files to /opt/input if enabled
    if parameters.get("read_local", False):
        local_input_dir = Path("/opt/input")
        
        parameters["raw_image_paths"] = [
            _copy_file_to_local(Path(p), local_input_dir) for p in image_paths
        ]
        parameters["raw_mask_paths"] = [
            _copy_file_to_local(Path(p), local_input_dir) for p in mask_paths
        ]
        parameters["local_input_dir"] = local_input_dir
    else:
        parameters["raw_image_paths"] = [Path(p) for p in image_paths]
        parameters["raw_mask_paths"] = [Path(p) for p in mask_paths]
        parameters["local_input_dir"] = None
    
    parameters["raw_image_names"] = [p.name for p in parameters["original_image_paths"]]
    parameters["raw_mask_names"] = [p.name for p in parameters["original_mask_paths"]]
    parameters["fragment_names"] = parameters["raw_image_names"]
    parameters["n_fragments"] = len(parameters["raw_image_paths"])
    parameters["resolution_scaling"] = [i / parameters["resolutions"][0] for i in parameters["resolutions"]]

    parameters["save_dir"].mkdir(parents=True, exist_ok=True)
    parameters["save_dir"].joinpath("configuration_detection", "checks").mkdir(parents=True, exist_ok=True)

    return parameters


def run_case(parameters):
    """
    Execute the full stitching pipeline for one case.
    """
    
    save_dir = parameters["save_dir"]
    output_res = parameters["output_res"]

    print(
        f"\nRunning job:"
        f"\n - Save dir: {save_dir}"
        f"\n - Output resolution: {output_res} µm/pixel\n"
    )

    log = _setup_logging(save_dir, parameters["my_level"])
    parameters["log"] = log
    log.log(parameters["my_level"], f"Running job with output resolution: {output_res} µm/pixel\n")

    prepare_data(parameters=parameters)
    solutions = detect_configuration(parameters=parameters)

    for count_sol, sol in enumerate(solutions, 1):
        log.log(parameters["my_level"], f"### Exploring solution {count_sol} ###")
        parameters["detected_configuration"] = sol
        parameters["num_sol"] = count_sol
        parameters["sol_save_dir"] = save_dir.joinpath(f"sol_{count_sol}")

        for count_res, res in enumerate(parameters["resolutions"]):
            parameters["iteration"] = count_res
            parameters["res_name"] = get_resname(res)
            parameters["fragment_names"] = [sol[i].lower() for i in sorted(sol)]

            fragments = [
                Fragment(im_path=im_path, fragment_name=fragment_name, kwargs=parameters)
                for im_path, fragment_name in sol.items()
            ]

            preprocess(fragments=fragments, parameters=parameters)
            optimize_stitch(parameters=parameters)

        generate_full_res(parameters=parameters, log=log)
        log.log(parameters["my_level"], f"### Succesfully stitched solution {count_sol} ###\n")

    log.log(parameters["my_level"], "PythoStitcher completed!")
    
    # Cleanup local input files if read_local was enabled
    if parameters.get("local_input_dir") is not None:
        _cleanup_local_directory(parameters["local_input_dir"])
    
    del parameters, log
    
    # Write per-case completion marker for downstream orchestration
    _write_completion_marker(save_dir)
    
    return


def parse_dataframe(df_path):
    """
    Parse CSV/XLSX and return validated cases grouped by save_path.
    """
    
    df = pd.read_csv(df_path) if df_path.suffix.lower() == ".csv" else pd.read_excel(df_path)
    
    required_cols = {"image_path", "mask_path", "save_path"}
    assert required_cols.issubset(set(df.columns)), "df must have columns image_path, mask_path, save_path"

    grouped = df.groupby("save_path")
    cases = []

    for save_path, group in grouped:
        ordered = group.sort_values("image_path")
        
        if len(ordered) not in [2, 4]:
            print(f"WARNING: case [{save_path}] has {len(ordered)} fragments; only 2 or 4 supported. Skipping.")
            continue

        image_paths = [Path(p) for p in ordered["image_path"]]
        mask_paths = [Path(p) for p in ordered["mask_path"]]

        if not all(p.exists() for p in image_paths + mask_paths):
            print(f"WARNING: Missing files for case [{Path(save_path).name}]; skipping.")
            continue

        # Scan for optional columns force_config and landmarks
        force_config_path = ordered["force_config_path"].dropna().iloc[0] if "force_config_path" in ordered.columns else None
        landmark_paths = [Path(p) for p in ordered["landmark_paths"]] if "landmark_paths" in ordered.columns else None

        cases.append({
            "image_paths": image_paths,
            "mask_paths": mask_paths,
            "save_path": Path(save_path),
            "force_config_path": force_config_path,
            "landmark_paths": landmark_paths,
        })

    return cases


def main():
    """
    Main function to run PythoStitcher using dataframe-driven batch.
    """
    
    parser = argparse.ArgumentParser(description="Stitch histopathology images into a pseudo whole-mount image")
    parser.add_argument("--df", required=True, type=Path, 
                        help="Path to a CSV/XLSX with columns: imagepath, maskpath, savepath, and optional force_config, landmark_path")
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to the parameter configuration JSON file")
    args = parser.parse_args()

    assert args.df.suffix.lower() in [".csv", ".xlsx"], "df must be .csv or .xlsx"
    assert args.df.exists(), f"df path {args.df} doesn't exist"
    assert args.config.exists(), f"config path {args.config} doesn't exist"
    assert args.config.suffix.lower() == ".json", "config must be a .json file"

    base_parameters = _load_base_parameters(args.config)
    assert "output_res" in base_parameters, "config must contain 'output_res' parameter"
    assert base_parameters["output_res"] > 0, "output resolution cannot be negative"

    cases = parse_dataframe(args.df)
    print(f"\n### Identified {len(cases)} cases. ###")

    for case in cases:
        parameters = load_parameters(
            image_paths=case["image_paths"], 
            mask_paths=case["mask_paths"], 
            save_path=case["save_path"], 
            force_config_path=case["force_config_path"],
            landmark_paths=case["landmark_paths"],
            base_parameters=base_parameters
        )
        try:
            run_case(parameters)
        except Exception as e:
            print(f"ERROR: Failed to run case {case['save_path']}: {e}")
            continue
        
    return


if __name__ == "__main__":
    main()
