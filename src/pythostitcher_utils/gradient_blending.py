import multiresolutionimageinterface as mir
import numpy as np
import cv2
import pyvips
import matplotlib.pyplot as plt
import time

from .fuse_images_highres import fuse_images_highres, is_valid_contour


def perform_blending(result_image, result_mask, full_res_fragments, log, parameters):
    """
    Function to blend areas of overlap using alpha blending.

    Inputs
        - Full resolution image
        - Full resolution mask
        - All fragments
        - Logging instance
        - Dictionary with parameters

    Output
        - Full resolution blended image
        - Computational time for blending
    """

    # Load .tif of the mask
    opener = mir.MultiResolutionImageReader()
    tif_mask = opener.open(parameters["tif_mask_path"])

    # Get output level closest to a 4k image
    best_mask_output_dims = 4000
    all_mask_dims = [tif_mask.getLevelDimensions(i) for i in range(tif_mask.getNumberOfLevels())]
    mask_ds_level = np.argmin([(i[0] - best_mask_output_dims) ** 2 for i in all_mask_dims])

    mask_ds = tif_mask.getUCharPatch(
        startX=0,
        startY=0,
        width=int(all_mask_dims[mask_ds_level][0]),
        height=int(all_mask_dims[mask_ds_level][1]),
        level=int(mask_ds_level),
    )

    # Get contour of overlapping areas and upsample to full resolution coords
    mask_cnts_ds, _ = cv2.findContours(
        (mask_ds == 2).astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    mask_cnts = [np.squeeze(i * (2 ** mask_ds_level)) for i in mask_cnts_ds]
    mask_cnts = [i for i in mask_cnts if is_valid_contour(i)]

    # Param for saving blending result
    n_valid = 0

    start = time.time()

    # Find blending points per contour
    for c, mask_cnt in enumerate(mask_cnts):

        log.log(45, f" - blending overlap area {c + 1}/{len(mask_cnts)}")

        # Get contour orientation and some starting variables
        cnt_dir = "hor" if np.std(mask_cnt[:, 0]) > np.std(mask_cnt[:, 1]) else "ver"
        long_end = 500
        max_image_width = all_mask_dims[0][0]
        max_image_height = all_mask_dims[0][1]

        # Get the length of the tile that needs to be stitched. Long end refers to the
        # direction of the contour while short end refers to the direction of the stitch.
        if cnt_dir == "hor":

            # Get limits of major axis which is horizontal
            long_end_start = np.min(mask_cnt[:, 0])
            long_end_end = np.max(mask_cnt[:, 0])

            # Get X coordinates spaced 500 pixels apart
            n_points = int(np.ceil((long_end_end - long_end_start) / long_end)) + 1
            cnt_points_x = list(np.linspace(long_end_start, long_end_end, n_points))
            cnt_points_x = list(map(int, cnt_points_x))
            cnt_points_x = np.array(
                [np.max([0, long_end_start - 50])]
                + cnt_points_x
                + [np.min([long_end_end + 50, max_image_width])]
            )

            # Draw a line along long axis and sample x coordinates
            short_end_start = mask_cnt[0, 1]
            short_end_end_idx = int(len(mask_cnt[:, 1]) / 2)
            short_end_end = mask_cnt[short_end_end_idx, 1]
            short_end_len = int((np.max(mask_cnt[:, 1]) - np.min(mask_cnt[:, 1])) * 2)
            cnt_points_y = np.linspace(short_end_start, short_end_end, len(cnt_points_x))
            cnt_points_y = cnt_points_y.astype("int")

        else:

            # Get limits of major axis which is vertical
            long_end_start = np.min(mask_cnt[:, 1])
            long_end_end = np.max(mask_cnt[:, 1])

            # Get Y coordinates spaced 500 pixels apart
            n_points = int(np.ceil((long_end_end - long_end_start) / long_end)) + 1
            cnt_points_y = list(np.linspace(long_end_start, long_end_end, n_points))
            cnt_points_y = list(map(int, cnt_points_y))
            cnt_points_y = np.array(
                [np.max([0, long_end_start - 50])]
                + cnt_points_y
                + [np.min([long_end_end + 50, max_image_height])]
            )

            # Draw a line along long axis and sample x coordinates
            short_end_start = mask_cnt[0, 0]
            short_end_end_idx = int(len(mask_cnt[:, 0]) / 2)
            short_end_end = mask_cnt[short_end_end_idx, 0]
            short_end_len = int((np.max(mask_cnt[:, 0]) - np.min(mask_cnt[:, 0])) * 2)
            cnt_points_x = np.linspace(short_end_start, short_end_end, len(cnt_points_y))
            cnt_points_x = cnt_points_x.astype("int")

        seed_points = np.vstack([cnt_points_x, cnt_points_y]).T

        # Blend per seed point
        for seed in seed_points:

            # Get tilesize and take into account not to cross image size borders
            if cnt_dir == "hor":
                xstart = seed[0]
                ystart = seed[1] - int(0.5 * (short_end_len))
                width = np.min([long_end, max_image_width - seed[0] - 1])
                height = np.min([short_end_len, max_image_height - seed[1] - 1])

            elif cnt_dir == "ver":
                xstart = seed[0] - int(0.5 * (short_end_len))
                ystart = seed[1]
                width = np.min([short_end_len, max_image_width - seed[0] - 1])
                height = np.min([long_end, max_image_height - seed[1] - 1])

            ### SANITY CHECK FOR TILE SELECTION
            # scale_factor = 2**mask_ds_level
            # xvals = [
            #     xstart/scale_factor,
            #     (xstart+width)/scale_factor,
            #     (xstart+width)/scale_factor,
            #     xstart / scale_factor,
            #     xstart / scale_factor
            #     ]
            # yvals = [
            #     ystart/scale_factor,
            #     ystart / scale_factor,
            #     (ystart + height) / scale_factor,
            #     (ystart + height) / scale_factor,
            #     ystart / scale_factor,
            # ]
            # plt.figure()
            # plt.imshow(mask_ds)
            # plt.plot(seed_points[:, 0]/8, seed_points[:, 1]/8, c="r")
            # plt.plot(xvals, yvals, c="r")
            # plt.show()

            # Only perform bending in case of overlap
            try:
                tile_mask = result_mask.crop(xstart, ystart, width, height)
            except:
                print("s")

            if tile_mask.max() > 1:

                # Extract the corresponding image and mask for all fragments
                images = dict()
                masks = dict()
                for f in full_res_fragments:
                    image_patch = f.final_image.crop(xstart, ystart, width, height)
                    image = np.ndarray(
                        buffer=image_patch.write_to_memory(),
                        dtype=np.uint8,
                        shape=[height, width, image_patch.bands],
                    )

                    mask_patch = f.outputres_mask.crop(xstart, ystart, width, height)
                    mask = np.ndarray(
                        buffer=mask_patch.write_to_memory(), dtype=np.uint8, shape=[height, width]
                    )

                    images[f.orientation] = image
                    masks[f.orientation] = mask

                # Perform the actual blending
                blend, grad, overlap_fragments, valid = fuse_images_highres(images, masks)

                if valid:

                    # Get overlap contours for plotting
                    overlap = (~np.isnan(grad) * 255).astype("uint8")
                    overlap_cnts, _ = cv2.findContours(
                        overlap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE,
                    )

                    # Show and save blended result
                    plt.figure(figsize=(12, 10))
                    plt.suptitle(f"Result at x {xstart} and y {ystart}", fontsize=24)
                    plt.subplot(231)
                    plt.title(f"Mask fragment '{overlap_fragments[0]}'", fontsize=20)
                    plt.imshow(masks[overlap_fragments[0]], cmap="gray")
                    plt.axis("off")
                    plt.clim([0, 1])
                    plt.subplot(232)
                    plt.title(f"Mask fragment '{overlap_fragments[1]}'", fontsize=20)
                    plt.imshow(masks[overlap_fragments[1]], cmap="gray")
                    plt.axis("off")
                    plt.clim([0, 1])
                    plt.subplot(233)
                    plt.title("Mask overlap + gradient", fontsize=20)
                    plt.imshow(
                        (masks[overlap_fragments[0]] + masks[overlap_fragments[1]]) == 2,
                        cmap="gray",
                    )
                    plt.imshow(grad, cmap="jet", alpha=0.5)
                    plt.axis("off")
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.subplot(234)
                    plt.title(f"Image fragment '{overlap_fragments[0]}'", fontsize=20)
                    plt.imshow(images[overlap_fragments[0]])
                    for cnt in overlap_cnts:
                        cnt = np.squeeze(cnt)
                        if len(cnt.shape) > 1:
                            plt.plot(cnt[:, 0], cnt[:, 1], c="r", linewidth=3)
                    plt.axis("off")
                    plt.subplot(235)
                    plt.title(f"Image fragment '{overlap_fragments[1]}'", fontsize=20)
                    plt.imshow(images[overlap_fragments[1]])
                    for cnt in overlap_cnts:
                        cnt = np.squeeze(cnt)
                        if len(cnt.shape) > 1:
                            plt.plot(cnt[:, 0], cnt[:, 1], c="r", linewidth=3)
                    plt.axis("off")
                    plt.subplot(236)
                    plt.title("Blend image", fontsize=20)
                    plt.imshow(blend)
                    for cnt in overlap_cnts:
                        cnt = np.squeeze(cnt)
                        if len(cnt.shape) > 1:
                            plt.plot(cnt[:, 0], cnt[:, 1], c="r", linewidth=3)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(
                        f"{parameters['blend_dir']}/contour{str(c).zfill(3)}_tile{str(n_valid).zfill(4)}.png"
                    )
                    plt.close()

                    n_valid += 1

                    # Insert blended image
                    h, w = blend.shape[:2]
                    bands = 3
                    dformat = "uchar"
                    blend_image = pyvips.Image.new_from_memory(blend.ravel(), w, h, bands, dformat)

                    result_image = result_image.insert(blend_image, xstart, ystart)

            else:
                continue

    comp_time = int(np.ceil((time.time() - start) / 60))

    # Prostates should be wider than they are high. If this is not the case, end result
    # may need to be rotated 90 degrees.
    y, x = np.nonzero(np.squeeze(mask_ds))
    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)
    if height > width:
        result_image = result_image.rotate(90)

    return result_image, comp_time
