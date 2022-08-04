import numpy as np
import pyvips
import cv2
import matplotlib.pyplot as plt
import os
import time
import glob
import logging

from utils.fuse_images_highres import fuse_images_highres


def create_smooth_mask(image):
    """
    Function to create a smooth mask to ensure a clean gradual transition between
    quadrants. When the mask is not smoothed the overlap boundary will have a lot
    of stray voxels which will induce artefacts in the final highres image.

    Input:
        - Image -> np.array

    Output:
        - Smoothed mask -> np.array
    """

    # Get initial mask based on tissue. We choose 3 as minimum threshold as there are
    # some noisy pixels with value 1/2/3 in 1 of the channels which we don't want to
    # include in our mask.
    mask = ((np.mean(image, axis=2) > 3) * 255).astype("uint8")

    # Smooth using 10x10 kernel
    mask = cv2.blur(mask, (10, 10))
    mask = (mask > 128) * 1

    return mask


def blend_image_tilewise(parameters, size, log):
    """
    Function to perform image blending per tile. These tiles are all saved in /tiles
    and can later be used by tile_based_reconstruction.py to reconstruct the final
    image. Some blending examples can be found in /blend_summary.

    Input:
        - All high resolution transformed quadrants

    Output:
        - Directory full of tiles with the blended quadrants.
    """

    start = time.time()
    log.critical("Blending quadrants")

    # Get original highres files from individual quadrants
    ul = pyvips.Image.new_from_file(
        f"../results/{parameters['patient_idx']}/highres/full_res_ul.tif"
    )
    ur = pyvips.Image.new_from_file(
        f"../results/{parameters['patient_idx']}/highres/full_res_ur.tif"
    )
    ll = pyvips.Image.new_from_file(
        f"../results/{parameters['patient_idx']}/highres/full_res_ll.tif"
    )
    lr = pyvips.Image.new_from_file(
        f"../results/{parameters['patient_idx']}/highres/full_res_lr.tif"
    )

    quadrants = [ul, ur, ll, lr]
    quadrant_names = ["ul", "ur", "ll", "lr"]
    tile_dir = f"../results/{parameters['patient_idx']}/highres/tiles"
    blend_dir = f"../results/{parameters['patient_idx']}/highres/blend_summary"

    for dir in [tile_dir, blend_dir]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    # Get dimensions of image
    width = ul.width
    height = ul.height

    # Specify tile sizes
    tilesize = [size] * 2
    num_xtiles = int(np.ceil(width / tilesize[0]))
    num_ytiles = int(np.ceil(height / tilesize[1]))

    # Loop over cols
    for x in range(num_xtiles):

        progress = int((x + 1) / num_xtiles * 100)
        if progress % 10 == 0 and progress != 0:
            log.critical(f" - progress {progress}%")

        # Loop over rows
        for y in range(num_ytiles):

            # To extract tiles via pyvips we need the starting X and Y and the tilesize.
            # This tilesize will differ based on whether we can retrieve a full square
            # tile, which is only not possible near the right and bottom edge. The tile
            # in these cases will be smaller.

            # Case for lower right corner
            if x == num_xtiles - 1 and y == num_ytiles - 1:
                new_tilesize = [width - x * tilesize[0], height - y * tilesize[1]]

            # Case for right edge
            elif x == num_xtiles - 1 and y < num_ytiles - 1:
                new_tilesize = [width - x * tilesize[0], tilesize[1]]

            # Case for bottom edge
            elif x < num_xtiles - 1 and y == num_ytiles - 1:
                new_tilesize = [tilesize[0], height - y * tilesize[1]]

            # Regular cases
            else:
                new_tilesize = tilesize

            patch_dict = dict()

            # Get tile of each quadrant. Height and width in the np.ndarray need to be
            # switched to get right size.
            for q, n in zip(quadrants, quadrant_names):
                patch = q.crop(
                    x * tilesize[0], y * tilesize[1], new_tilesize[0], new_tilesize[1]
                )
                patch_np = np.ndarray(
                    buffer=patch.write_to_memory(),
                    dtype=np.uint8,
                    shape=[new_tilesize[1], new_tilesize[0], patch.bands],
                )
                patch_dict[n] = patch_np

            # Smooth the masks a bit and apply this to the images for consistency
            images = list(patch_dict.values())
            masks = [create_smooth_mask(im) for im in images]
            images = [i * m[:, :, np.newaxis] for i, m in zip(images, masks)]

            # Check if there is any overlap between quadrants
            is_overlap = (np.sum(masks, axis=0) > 1).any()

            # If so, use alpha blending
            if is_overlap:
                blend, grad, overlap_quadrants, angle, valid = fuse_images_highres(
                    images, masks
                )

                # Valid indicates that the overlap was significant enough to perform
                # alpha blending. If only a few pixels overlap, we do not perform alpha
                # blending.
                if valid:

                    # Show and save blended result
                    plt.figure(figsize=(12, 10))
                    plt.suptitle(f"Result for row {y} and col {x}", fontsize=24)
                    plt.subplot(231)
                    plt.title("Mask Q1", fontsize=20)
                    plt.imshow(masks[overlap_quadrants[0]], cmap="gray")
                    plt.axis("off")
                    plt.clim([0, 1])
                    plt.subplot(232)
                    plt.title("Mask Q2", fontsize=20)
                    plt.imshow(masks[overlap_quadrants[1]], cmap="gray")
                    plt.axis("off")
                    plt.clim([0, 1])
                    plt.subplot(233)
                    plt.title("Mask overlap + gradient", fontsize=20)
                    plt.imshow(
                        (masks[overlap_quadrants[0]] + masks[overlap_quadrants[1]]) == 2,
                        cmap="gray",
                    )
                    plt.imshow(grad, cmap="jet", alpha=0.5)
                    plt.axis("off")
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.subplot(234)
                    plt.title("Image Q1", fontsize=20)
                    plt.imshow(images[overlap_quadrants[0]])
                    plt.axis("off")
                    plt.subplot(235)
                    plt.title("Image Q2", fontsize=20)
                    plt.imshow(images[overlap_quadrants[1]])
                    plt.axis("off")
                    plt.subplot(236)
                    plt.title("Blend image", fontsize=20)
                    plt.imshow(blend)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(
                        f"{blend_dir}/row{str(y).zfill(4)}_col{str(x).zfill(4)}.png"
                    )
                    plt.close()

            # In case of no overlap we can simply sum the images
            else:
                blend = np.sum(images, axis=0).astype("uint8")

            # Save blended tile
            cv2.imwrite(
                f"{tile_dir}/row{str(y).zfill(4)}_col{str(x).zfill(4)}.png",
                cv2.cvtColor(blend, cv2.COLOR_RGB2BGR),
            )

    # Save small dict with some values for reconstruction
    log.critical(
        f" > finished creating tiles in {int((time.time() - start)/60)} mins!\n"
    )

    with open(f"{tile_dir}/info.txt", "w") as f:
        f.write(f"cols:{num_xtiles}\n")
        f.write(f"rows:{num_ytiles}\n")
        f.write(f"tilesize_width:{tilesize[0]}\n")
        f.write(f"tilesize_height:{tilesize[1]}\n")
        f.write(f"targetsize_width:{width}\n")
        f.write(f"targetsize_height:{height}\n")
    f.close()

    return
