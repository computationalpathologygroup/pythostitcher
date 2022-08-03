import os
import numpy as np
import pyvips
import glob
import time


def eval_handler(full_image, progress):
    """
    Function to track progress of a pyvips operation. Name of variable to be tracked
    must be included as the first argument.
    """

    print(
        f"  > elapsed time: {int(progress.run/60)} min, progress: {progress.percent}%"
    )
    time.sleep(60)

    return


def reconstruct_image(parameters):
    """
    Function to reconstruct the high resolution image by assembling all the previously
    created tiles from the blend_image_tilewise function.

    Input:
        - Dict with parameters
        - Directory of tiles

    Output:
        - Reconstructed blended image at full resolution
    """

    basedir = f"../results/{parameters['patient_idx']}/highres"
    tile_dir = f"../results/{parameters['patient_idx']}/highres/tiles"
    tif_dir = f"../results/{parameters['patient_idx']}/highres/temp_tifs"

    if not os.path.isdir(tif_dir):
        os.mkdir(tif_dir)

    tile_info = np.load(f"{tile_dir}/info.npy", allow_pickle=True).item()

    print("\nCreating tile-based reconstruction")

    # Loop over all rows
    for row in range(tile_info["rows"]):
        print(
            f"Processing row {str(row).zfill(len(str(tile_info['rows'])))}/{tile_info['rows']}"
        )

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

    # Get all tifs and assemble final image
    print(f"\nSaving fullres image")
    tif_filenames = sorted(glob.glob(f"{tif_dir}/*"))
    tif_files = [pyvips.Image.new_from_file(t) for t in tif_filenames]
    full_image = pyvips.Image.arrayjoin(tif_files, across=1)

    # Save final image
    full_image.set_progress(True)
    full_image.signal_connect("eval", eval_handler)
    full_image.tiffsave(
        f"../results/{parameters['patient_idx']}/highres/full_res_blended_image.tif",
        tile=True,
        compression="jpeg",
        bigtiff=True,
        pyramid=True,
        Q=90,
    )

    return
