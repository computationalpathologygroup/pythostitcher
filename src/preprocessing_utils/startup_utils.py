import os
import json
import logging
import shutil
from pathlib import Path

import pyvips


def _configure_pyvips(params):
    """
    Optionally configure pyvips runtime from JSON config.
    """
    cfg = params.get("pyvips", {}) or {}
    if cfg.get("cache_max") is not None:
        pyvips.cache_set_max(int(cfg["cache_max"]))
    if cfg.get("cache_max_files") is not None:
        pyvips.cache_set_max_files(int(cfg["cache_max_files"]))
    if cfg.get("cache_max_mem_gb") is not None:
        pyvips.cache_set_max_mem(int(float(cfg["cache_max_mem_gb"]) * (1024 ** 3)))
    if cfg.get("cache_trace", False):
        pyvips.cache_set_trace(True)
    print(
        f"pyvips configured -> "
        f"concurrency={os.environ['VIPS_CONCURRENCY']}, "
        f"cache_max={pyvips.cache_get_max()}, "
        f"cache_max_files={pyvips.cache_get_max_files()}, "
        f"cache_max_mem={pyvips.cache_get_max_mem()} bytes"
    )


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


