import pyvips
import multiresolutionimageinterface as mir
import numpy as np
import cv2
import time
import os
import logging
import copy

from .get_resname import get_resname


def eval_handler_image(final_masked_image, progress):
    """
    Function to display progress of pyvips operation.
    """

    if progress.percent % 10 == 0 and progress.percent != 0:
        handler_log.critical(f" - progress {progress.percent}%")

    time.sleep(10)

    return


def eval_handler_combo(fullres_combo, progress):
    """
    Function to display progress of pyvips operation.
    """

    if progress.percent % 10 == 0 and progress.percent != 0:
        handler_log.critical(f" - progress {progress.percent}%")

    time.sleep(10)

    return


def write_highres_quadrants(parameters, log, sanity_check):
    """
    Function to write a full resolution output image of the transformed
    quadrant. This output image will only contain one specific quadrant. This quadrant
    can then serve as input for the final blending algorithm. The outline of this
    function is as follows:
    1. Obtain high resolution mask
    2. Preprocess the mask
    3. Preprocess the image
    4. Apply masking to image
    5. Apply transformation to image

    Input:
        - Dict with parameters
        - Original mask
        - Original full resolution image (.mrxs)

    Output
        - Full resolution .tif image of transformed quadrant
    """

    opener = mir.MultiResolutionImageReader()
    quadrants = ["ul", "ur", "ll", "lr"]

    for q in quadrants:

        # Check if file already exists
        savefile = f"../results/{parameters['patient_idx']}/highres/full_res_{q}.tif"

        if not os.path.isfile(savefile):

            start = time.time()
            log.critical(f"Generating full resolution quadrant {q.upper()}")

            # Get original mask and image
            mask = opener.open(
                f"../sample_data/{parameters['patient_idx']}/raw_masks/{q}.tif"
            )
            image = opener.open(
                f"../sample_data/{parameters['patient_idx']}/raw_images/{q}.mrxs"
            )

            # ===========================================================================
            # SCALE CONVERSIONS AND QUADRANT INFO
            # ===========================================================================

            # Get dimensions of mask and image
            mask_dims = mask.getLevelDimensions(0)
            image_dims = image.getLevelDimensions(0)
            scaling_mask2fullres = int(image_dims[0] / mask_dims[0])

            # Get shape of preprocessed mask from pythostitcher
            resname = get_resname(parameters["resolutions"][-1])
            ps_highest_res_mask = cv2.imread(
                f"../results/{parameters['patient_idx']}/{parameters['slice_idx']}/"
                + f"{resname}/quadrant_{q.upper()}_mask.png"
            )
            ps_highest_res_mask = cv2.cvtColor(ps_highest_res_mask, cv2.COLOR_BGR2GRAY)

            # Get upsample factor to get to full resolution. This upsample factor consists
            # of 2 components: 1) the level to which the original image was downsampled to
            # make the image suitable for pythostitcher and 2) the fraction of the
            # resolution in 1) that was used as the highest resolution in pythostitcher.
            ps_level = parameters["image_level"]
            scaling_ps2fullres = int(
                image.getLevelDownsample(ps_level) * 1 / (parameters["resolutions"][-1])
            )
            target_dims = [
                int(i * scaling_ps2fullres) for i in ps_highest_res_mask.shape
            ]

            # Get the highest resolution tform from pythostitcher
            ps_tform = np.load(
                f"../results/{parameters['patient_idx']}/tform/res0500_tform_final.npy",
                allow_pickle=True,
            ).item()

            # Get upsampled pythostitcher transformation. This scales linearly with
            # resolution.
            highres_tform = [
                int(ps_tform[q.upper()][0] * scaling_ps2fullres),
                int(ps_tform[q.upper()][1] * scaling_ps2fullres),
                np.round(ps_tform[q.upper()][2], 1),
                tuple([int(i * scaling_ps2fullres) for i in ps_tform[q.upper()][3]]),
                tuple([int(i * scaling_ps2fullres) for i in ps_tform[q.upper()][4]]),
            ]

            # Get manually obtained rotation
            with open(f"../sample_data/{parameters['patient_idx']}/rotations.txt") as f:
                lines = []
                for line in f:
                    line = line.split()
                    lines.extend(line)
            f.close()

            q_idx_line = np.argmax([q.upper() in i for i in lines])
            q_line = lines[q_idx_line]
            hflip, vflip = False, False

            if "hf" in q_line:
                hflip = True
                q_line = q_line.replace("hf", "")
            if "vf" in q_line:
                vflip = True
                q_line = q_line.replace("vf", "")

            angle = int(q_line.split(":")[-1])
            angle_k = int(angle / 90)

            # ===========================================================================
            # MASK PREPROCESSING
            # ===========================================================================

            # Get high resolution mask (spacing 3.88x3.88)
            original_mask = mask.getUCharPatch(
                startX=0, startY=0, width=mask_dims[0], height=mask_dims[1], level=0
            )

            # Convert mask for opencv processing
            original_mask = original_mask / np.max(original_mask)
            original_mask = (original_mask * 255).astype("uint8")

            # Get information on all connected components in the mask
            num_labels, labeled_mask, stats, _ = cv2.connectedComponentsWithStats(
                original_mask, connectivity=4
            )

            # Extract largest connected component
            largest_cc_label = np.argmax(stats[1:, -1]) + 1
            original_mask = ((labeled_mask == largest_cc_label) * 255).astype("uint8")

            # Some morphological operations for cleaning up edges
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(40, 40))
            original_mask = cv2.morphologyEx(
                src=original_mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=2
            )

            # Flood fill to remove holes inside mask. The floodfill mask is required
            # by opencv
            seedpoint = (0, 0)
            floodfill_mask = np.zeros(
                (original_mask.shape[0] + 2, original_mask.shape[1] + 2)
            ).astype("uint8")
            _, _, original_mask, _ = cv2.floodFill(
                original_mask, floodfill_mask, seedpoint, 255
            )
            original_mask = 1 - original_mask[1:-1, 1:-1]

            # Get nonzero indices and crop
            r, c = np.nonzero(original_mask)
            original_mask = original_mask[np.min(r) : np.max(r), np.min(c) : np.max(c)]

            # Rotate and flip if necessary
            original_mask = np.rot90(original_mask, k=angle_k)
            if hflip:
                original_mask = np.fliplr(original_mask)
            if vflip:
                original_mask = np.flipud(original_mask)

            # Convert to pyvips array
            height, width = original_mask.shape
            bands = 1
            dformat = "uchar"
            fullres_mask = pyvips.Image.new_from_memory(
                original_mask.ravel(), width, height, bands, dformat
            )

            fullres_mask = fullres_mask.resize(scaling_mask2fullres)

            # Pad image with zeros
            fullres_mask = fullres_mask.gravity(
                "centre", target_dims[1], target_dims[0]
            )

            # ===========================================================================
            # PREPARE FULL RES IMAGE
            # ===========================================================================

            fullres_image = pyvips.Image.new_from_file(
                f"../sample_data/{parameters['patient_idx']}/raw_images/{q}.mrxs"
            )

            # Dispose of alpha channel if applicable
            if fullres_image.hasalpha():
                fullres_image = fullres_image.flatten()

            # Get cropping indices
            rmin, rmax = (
                int(scaling_mask2fullres * np.min(r)),
                int(scaling_mask2fullres * np.max(r)),
            )
            cmin, cmax = (
                int(scaling_mask2fullres * np.min(c)),
                int(scaling_mask2fullres * np.max(c)),
            )
            width = cmax - cmin
            height = rmax - rmin

            # Crop image
            fullres_image = fullres_image.crop(cmin, rmin, width, height)

            # Rotate image
            fullres_image = fullres_image.rotate(-angle)

            # Flip if necessary
            if hflip:
                fullres_image = fullres_image.fliphor()
            if vflip:
                fullres_image = fullres_image.flipver()

            # Pad image with zeros
            fullres_image = fullres_image.gravity(
                "centre", target_dims[1], target_dims[0]
            )

            if not fullres_image.format == "uchar":
                fullres_image = fullres_image.cast("uchar", shift=False)

            # ===========================================================================
            # PREPARE FULL RES MASKED IMAGE
            # ===========================================================================

            # Apply mask to image
            final_image = fullres_image.multiply(fullres_mask)

            # Get transformation matrix
            rotmat = cv2.getRotationMatrix2D(
                center=highres_tform[3], angle=highres_tform[2], scale=1
            )
            rotmat[0, 2] += highres_tform[0]
            rotmat[1, 2] += highres_tform[1]

            # Apply transformation to masked image
            final_masked_image = final_image.affine(
                (rotmat[0, 0], rotmat[0, 1], rotmat[1, 0], rotmat[1, 1]),
                interpolate=pyvips.Interpolate.new("nearest"),
                odx=rotmat[0, 2],
                ody=rotmat[1, 2],
                oarea=[0, 0, highres_tform[4][1], highres_tform[4][0]],
            )

            if final_masked_image.hasalpha():
                final_masked_image = final_masked_image.flatten()

            if not final_masked_image.format == "uchar":
                final_masked_image = final_masked_image.cast("uchar", shift=False)

            # Save final masked image.
            # NOTE: JPEG QUALITY IS VERY LOW JUST TO SAVE WRITING TIME
            global handler_log
            handler_log = copy.copy(log)
            final_masked_image.set_progress(True)
            final_masked_image.signal_connect("eval", eval_handler_image)
            final_masked_image.tiffsave(
                savefile,
                tile=True,
                compression="jpeg",
                bigtiff=True,
                pyramid=True,
                Q=25,
            )

            log.critical(
                f" > total processing time {int((time.time() - start) / 60)} mins\n"
            )

    if sanity_check:
        # Sum all images as a very coarse "blending". This only serves as a sanity check
        # to ensure that all transformations were performed correctly and this will
        # be deleted in the next stable version.
        fullres_ul = pyvips.Image.new_from_file(
            f"../results/{parameters['patient_idx']}/highres/full_res_ul.tif"
        )
        fullres_ur = pyvips.Image.new_from_file(
            f"../results/{parameters['patient_idx']}/highres/full_res_ur.tif"
        )
        fullres_lr = pyvips.Image.new_from_file(
            f"../results/{parameters['patient_idx']}/highres/full_res_lr.tif"
        )
        fullres_ll = pyvips.Image.new_from_file(
            f"../results/{parameters['patient_idx']}/highres/full_res_ll.tif"
        )

        log.critical("Computing recombined image")
        start = time.time()

        fullres_upper = fullres_ul.add(fullres_ur)
        fullres_lower = fullres_ll.add(fullres_lr)
        fullres_combo = fullres_upper.add(fullres_lower)

        # Dispose of alpha channel
        if fullres_combo.hasalpha():
            fullres_combo = fullres_combo.flatten()

        if not fullres_combo.format == "uchar":
            fullres_combo = fullres_combo.cast("uchar", shift=False)

        # NOTE: JPEG QUALITY IS VERY LOW JUST TO SAVE WRITING TIME
        fullres_combo.set_progress(True)
        fullres_combo.signal_connect("eval", eval_handler_combo)
        fullres_combo.tiffsave(
            f"../results/{parameters['patient_idx']}/highres/full_res_added.tif",
            tile=True,
            compression="jpeg",
            bigtiff=True,
            pyramid=True,
            Q=25,
        )

        log.critical(f" > finished saving image in {int((time.time()-start)/60)} mins")

    return
