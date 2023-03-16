import multiresolutionimageinterface as mir
import numpy as np
import cv2
import pyvips
import matplotlib.pyplot as plt


def perform_blending(result_image, result_mask, full_res_fragments, log, parameters):
    """
    Function to blend areas of overlap.

    Inputs
        - Full resolution image
        - Full resolution mask
        - All fragments
        - Logging instance
        - Directory to save blending results

    Output
        - Full resolution blended image
    """

    # Get dimensions of image
    width = result_mask.width
    height = result_mask.height

    # Specify tile sizes
    tilesize = 4000
    num_xtiles = int(np.ceil(width / tilesize))
    num_ytiles = int(np.ceil(height / tilesize))

    start = time.time()

    # Loop over columns
    for x in range(num_xtiles):

        log.log(45, f" - blending column {x+1}/{num_xtiles}")

        # Loop over rows
        for y in range(num_ytiles):

            # To extract tiles via pyvips we need the starting X and Y and the tilesize.
            # This tilesize will differ based on whether we can retrieve a full square
            # tile, which is only not possible near the right and bottom edge. The tile
            # in these cases will be smaller.

            # Case for lower right corner
            if x == num_xtiles - 1 and y == num_ytiles - 1:
                new_tilesize = [width - x * tilesize, height - y * tilesize]
            # Case for right edge
            elif x == num_xtiles - 1 and y < num_ytiles - 1:
                new_tilesize = [width - x * tilesize, tilesize]
            # Case for bottom edge
            elif x < num_xtiles - 1 and y == num_ytiles - 1:
                new_tilesize = [tilesize, height - y * tilesize]
            # Regular cases
            else:
                new_tilesize = [tilesize, tilesize]

            # Only perform bending in case of overlap
            tile_mask = result_mask.crop(
                x * tilesize, y * tilesize, new_tilesize[0], new_tilesize[1]
            )

            if tile_mask.max() > 1:

                # Extract the corresponding image and mask for all fragments
                images = dict()
                masks = dict()
                for f in full_res_fragments:
                    image_patch = f.outputres_mask.crop(
                        x * tilesize, y * tilesize, new_tilesize[0], new_tilesize[1],
                    )
                    image = np.ndarray(
                        buffer=image_patch.write_to_memory(),
                        dtype=np.uint8,
                        shape=[new_tilesize[1], new_tilesize[0], image_patch.bands],
                    )

                    mask_patch = f.outputres_mask.crop(
                        x * tilesize, y * tilesize, new_tilesize[0], new_tilesize[1],
                    )
                    mask = np.ndarray(
                        buffer=mask_patch.write_to_memory(),
                        dtype=np.uint8,
                        shape=[new_tilesize[1], new_tilesize[0]],
                    )

                    images[f.orientation] = image
                    masks[f.orientation] = mask

                # Perform the actual blending
                blend, grad, overlap_fragments, valid = fuse_images_highres(
                    images, masks
                )

                if valid:

                    # Get overlap contours for plotting
                    overlap = (~np.isnan(grad) * 255).astype("uint8")
                    overlap_cnts, _ = cv2.findContours(
                        overlap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE,
                    )

                    # Show and save blended result
                    plt.figure(figsize=(12, 10))
                    plt.suptitle(f"Result for row {y} and col {x}", fontsize=24)
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
                        (masks[overlap_fragments[0]] + masks[overlap_fragments[1]])==2,
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
                        plt.plot(cnt[:, 0], cnt[:, 1], c="r", linewidth=3)
                    plt.axis("off")
                    plt.subplot(235)
                    plt.title(f"Image fragment '{overlap_fragments[1]}'", fontsize=20)
                    plt.imshow(images[overlap_fragments[1]])
                    for cnt in overlap_cnts:
                        cnt = np.squeeze(cnt)
                        plt.plot(cnt[:, 0], cnt[:, 1], c="r", linewidth=3)
                    plt.axis("off")
                    plt.subplot(236)
                    plt.title("Blend image", fontsize=20)
                    plt.imshow(blend)
                    for cnt in overlap_cnts:
                        cnt = np.squeeze(cnt)
                        plt.plot(cnt[:, 0], cnt[:, 1], c="r", linewidth=3)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(
                        f"{parameters['blend_dir']}/row{str(y).zfill(4)}_col{str(x).zfill(4)}.png"
                    )
                    plt.close()

                    # Insert blended image
                    h, w = blend.shape[:2]
                    bands = 3
                    dformat = "uchar"
                    blend_image = pyvips.Image.new_from_memory(
                        blend.ravel(), w, h, bands, dformat
                    )

                    result_image = result_image.insert(
                        blend_image, x * new_tilesize[0], y * new_tilesize[1]
                    )

            else:
                continue

    comp_time = int(np.ceil((time.time() - start) / 60))

    return result_image, comp_time