import os
import numpy as np
import pyvips
import glob
import time
import logging


def eval_handler(full_image, progress):
    """
    Function to track progress of a pyvips operation. Name of variable to be tracked
    must be included as the first argument.
    """

    if progress.percent % 10 == 0 and progress.percent != 0:
        handler_log.critical(f" - progress {int(progress.percent)}%")

    time.sleep(10)

    return


def reconstruct_image(parameters, log):
    """
    Function to reconstruct the high resolution image by assembling all the previously
    created tiles from the blend_image_tilewise function.

    Input:
        - Dict with parameters
        - Directory of tiles

    Output:
        - Reconstructed blended image at full resolution
    """

    basedir = f"{parameters['results_dir']}/highres"
    tile_dir = f"{parameters['results_dir']}/highres/tiles"
    tif_dir = f"{parameters['results_dir']}/highres/temp_tifs"

    if not os.path.isdir(tif_dir):
        os.mkdir(tif_dir)

    # Get required reconstruction info
    with open(f"{parameters['results_dir']}/highres/tiles/info.txt") as f:
        lines = []
        for line in f:
            line = line.split()
            lines.extend(line)
        f.close()

    tile_info = dict()
    for line in lines:
        key, val = line.split(":")
        tile_info[key] = int(val)

    log.critical("Performing tile-based reconstruction")
    start = time.time()

    # Loop over all rows
    for row in range(tile_info["rows"]):

        progress = int((row + 1) / tile_info["rows"] * 100)
        if progress % 10 == 0 and progress != 0:
            log.critical(f" - progress {progress}%")

        images = []

        # Concatenate all column images of a certain row
        for col in range(tile_info["cols"]):
            im = pyvips.Image.new_from_file(
                f"{tile_dir}/row{str(row).zfill(4)}_col{str(col).zfill(4)}.png"
            )

            images.append(im)

        # Save resulting row as tif. If we try to reconstruct the final image by keeping
        # all these rows in the memory, we will run out of memory.
        full_row = pyvips.Image.arrayjoin(images, across=len(images))
        full_row.tiffsave(
            f"{tif_dir}/row_{str(row).zfill(4)}.tif",
            tile=True,
            compression="jpeg",
            Q=90,
        )

    log.critical(
        f" > finished assembling tiles in {int((time.time() - start)/60)} mins!\n"
    )

    # Get all tifs and assemble final image
    log.critical(f"Saving final image")
    tif_filenames = sorted(glob.glob(f"{tif_dir}/*"))
    tif_files = [pyvips.Image.new_from_file(t) for t in tif_filenames]
    full_image = pyvips.Image.arrayjoin(tif_files, across=1)

    # Save final image
    global handler_log
    handler_log = copy.copy(log)
    start = time.time()
    full_image.set_progress(True)
    full_image.signal_connect("eval", eval_handler)
    full_image.tiffsave(
        f"{parameters['results_dir']}/highres/full_res_blended_image.tif",
        tile=True,
        compression="jpeg",
        bigtiff=True,
        pyramid=True,
        Q=90,
    )
    log.critical(f" > finished saving image in {int((time.time() - start)/60)} mins!\n")

    return
