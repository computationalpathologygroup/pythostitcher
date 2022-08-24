import numpy as np
import os
import pyvips
import cv2
import multiresolutionimageinterface as mir
import time
import copy
import matplotlib.pyplot as plt

from .get_resname import get_resname
from .fuse_images_highres import fuse_images_highres


class FullResImage:
    """
    Class for the full resolution fragments. This class contains several methods to
    process the full resolution fragments based on the transformation obtained by
    PythoStitcher.
    """

    def __init__(self, parameters, name):

        self.name = name
        self.tissue = parameters["tissue"]
        self.last_res = parameters["resolutions"][-1]
        self.res_name = get_resname(self.last_res)
        self.slice_idx = parameters["slice_idx"]
        self.ps_level = parameters["image_level"]

        self.results_dir = parameters["results_dir"]
        self.data_dir = parameters["data_dir"]
        self.raw_mask_path = f"{self.data_dir}/raw_masks/{self.name}.tif"
        if os.path.isfile(f"{self.data_dir}/raw_images/{self.name}.tif"):
            self.raw_image_path = f"{self.data_dir}/raw_images/{self.name}.tif"
        elif os.path.isfile(f"{self.data_dir}/raw_images/{self.name}.mrxs"):
            self.raw_image_path = f"{self.data_dir}/raw_images/{self.name}.mrxs"
        self.ps_mask_path = (
            f"{self.results_dir}/images/{self.slice_idx}/{self.res_name}/"
            f"fragment_{self.name}_mask.png"
        )

        return

    def load_images(self):
        """
        Load the raw image, raw mask and the PythoStitcher mask from the highest
        resolution.
        """

        self.raw_image = pyvips.Image.new_from_file(self.raw_image_path)
        self.raw_image_dims = (
            self.raw_image.get("width"),
            self.raw_image.get("height"),
        )

        self.raw_mask = pyvips.Image.new_from_file(self.raw_mask_path)
        self.raw_mask_dims = (self.raw_mask.get("width"), self.raw_mask.get("height"))

        self.ps_mask = cv2.imread(self.ps_mask_path)
        self.ps_mask = cv2.cvtColor(self.ps_mask, cv2.COLOR_BGR2GRAY)

        return

    def get_scaling(self):
        """
        Get scaling factors to go from PythoStitcher resolution to full resolution
        """

        # Get scaling factor between PythoStitcher mask, raw mask and raw image
        self.scaling_mask2fullres = int(self.raw_image_dims[0] / self.raw_mask_dims[0])
        if self.tissue == "prostate":
            self.scaling_ps2fullres = int(
                int(self.raw_image.get(f"openslide.level[{self.ps_level}].downsample"))
                / self.last_res
            )
        elif self.tissue == "oesophagus":
            opener = mir.MultiResolutionImageReader()
            _raw_image = opener.open(self.raw_image_path)
            self.scaling_ps2fullres = (
                int(_raw_image.getLevelDownsample(self.ps_level)) / self.last_res
            )

        # Dimension of final stitchting result
        self.target_dims = [
            int(i * self.scaling_ps2fullres) for i in self.ps_mask.shape
        ]

        # Get the optimal transformation obtained with the genetic algorithm
        self.ps_tform = np.load(
            f"{self.results_dir}/tform/{self.res_name}_tform_final.npy",
            allow_pickle=True,
        ).item()

        # Upsample it to use it for the final image
        self.highres_tform = [
            int(self.ps_tform[self.name][0] * self.scaling_ps2fullres),
            int(self.ps_tform[self.name][1] * self.scaling_ps2fullres),
            np.round(self.ps_tform[self.name][2], 1),
            tuple(
                [int(i * self.scaling_ps2fullres) for i in self.ps_tform[self.name][3]]
            ),
            tuple(
                [int(i * self.scaling_ps2fullres) for i in self.ps_tform[self.name][4]]
            ),
        ]

        return

    def get_rotation(self):
        """
        Retrieve the original manually specified rotation.
        """

        # Get file with the specified rotations
        with open(f"{self.data_dir}/rotations.txt") as rot_file:
            lines = rot_file.read().splitlines()

        # Get the line with the current fragment
        f_idx_line = np.argmax(
            [((self.name in i) or (self.name.upper() in i)) for i in lines]
        )
        f_line = lines[f_idx_line]
        self.hflip, self.vflip = False, False

        # Check if fragment needs to be flipped horizontally/vertically
        if "hf" in f_line:
            self.hflip = True
            f_line = f_line.replace("hf", "")
        if "vf" in f_line:
            self.vflip = True
            f_line = f_line.replace("vf", "")

        # Get the rotation angle
        self.angle = int(f_line.split(":")[-1])
        self.angle_k = int(self.angle / 90)

        return

    def process_mask(self):
        """
        Process the mask to a full resolution version
        """

        # Get high resolution mask (spacing 3.88x3.88)
        opener = mir.MultiResolutionImageReader()
        mask = opener.open(self.raw_mask_path)
        original_mask = mask.getUCharPatch(
            startX=0,
            startY=0,
            width=self.raw_mask_dims[0],
            height=self.raw_mask_dims[1],
            level=0,
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

        # Slightly enlarge temporary
        temp_pad = int(0.05 * original_mask.shape[0])
        original_mask = np.pad(
            original_mask,
            [[temp_pad, temp_pad], [temp_pad, temp_pad]],
            mode="constant",
            constant_values=0,
        )

        # Flood fill to remove holes inside mask
        seedpoint = (0, 0)
        floodfill_mask = np.zeros(
            (original_mask.shape[0] + 2, original_mask.shape[1] + 2)
        ).astype("uint8")
        _, _, original_mask, _ = cv2.floodFill(
            original_mask, floodfill_mask, seedpoint, 255
        )
        original_mask = (
            1 - original_mask[temp_pad+1:-(temp_pad+1), temp_pad+1:-(temp_pad+1)]
        )

        # Get nonzero indices and crop
        self.r_idx, self.c_idx = np.nonzero(original_mask)
        original_mask = original_mask[
            np.min(self.r_idx) : np.max(self.r_idx),
            np.min(self.c_idx) : np.max(self.c_idx),
        ]

        # Rotate and flip if necessary
        original_mask = np.rot90(original_mask, k=self.angle_k)
        if self.hflip:
            original_mask = np.fliplr(original_mask)
        if self.vflip:
            original_mask = np.flipud(original_mask)

        # Convert to pyvips array
        height, width = original_mask.shape
        bands = 1
        dformat = "uchar"
        self.fullres_mask = pyvips.Image.new_from_memory(
            original_mask.ravel(), width, height, bands, dformat
        )

        self.fullres_mask = self.fullres_mask.resize(self.scaling_mask2fullres)

        # Pad image with zeros
        self.fullres_mask = self.fullres_mask.gravity(
            "centre", self.target_dims[1], self.target_dims[0]
        )

        return

    def process_image(self):
        """
        Process the raw image to the full resolution version
        """

        # Dispose of alpha channel if applicable
        if self.raw_image.hasalpha():
            self.raw_image = self.raw_image.flatten()

        # Get cropping indices
        rmin, rmax = (
            int(self.scaling_mask2fullres * np.min(self.r_idx)),
            int(self.scaling_mask2fullres * np.max(self.r_idx)),
        )
        cmin, cmax = (
            int(self.scaling_mask2fullres * np.min(self.c_idx)),
            int(self.scaling_mask2fullres * np.max(self.c_idx)),
        )
        width = cmax - cmin
        height = rmax - rmin

        # Crop image
        self.fullres_image = self.raw_image.crop(cmin, rmin, width, height)

        # Rotate image
        self.fullres_image = self.fullres_image.rotate(-self.angle)

        # Flip if necessary
        if self.hflip:
            self.fullres_image = self.fullres_image.fliphor()
        if self.vflip:
            sefl.fullres_image = self.fullres_image.flipver()

        # Pad image with zeros
        self.fullres_image = self.fullres_image.gravity(
            "centre", self.target_dims[1], self.target_dims[0]
        )

        # Get transformation matrix
        rotmat = cv2.getRotationMatrix2D(
            center=self.highres_tform[3], angle=self.highres_tform[2], scale=1
        )
        rotmat[0, 2] += self.highres_tform[0]
        rotmat[1, 2] += self.highres_tform[1]

        # Apply affine transformation
        self.fullres_image_rot = self.fullres_image.affine(
            (rotmat[0, 0], rotmat[0, 1], rotmat[1, 0], rotmat[1, 1]),
            interpolate=pyvips.Interpolate.new("nearest"),
            odx=rotmat[0, 2],
            ody=rotmat[1, 2],
            oarea=[0, 0, self.highres_tform[4][1], self.highres_tform[4][0]],
        )

        self.fullres_mask_rot = self.fullres_mask.affine(
            (rotmat[0, 0], rotmat[0, 1], rotmat[1, 0], rotmat[1, 1]),
            interpolate=pyvips.Interpolate.new("nearest"),
            odx=rotmat[0, 2],
            ody=rotmat[1, 2],
            oarea=[0, 0, self.highres_tform[4][1], self.highres_tform[4][0]],
        )

        if not self.fullres_image_rot.format == "uchar":
            self.fullres_image_rot = self.fullres_image_rot.cast("uchar", shift=False)

        if not self.fullres_mask_rot.format == "uchar":
            self.fullres_mask_rot = self.fullres_mask_rot.cast("uchar", shift=False)

        # Apply mask to images
        self.final_image = self.fullres_image_rot.multiply(self.fullres_mask_rot)
        if not self.final_image.format == "uchar":
            self.final_image = self.final_image.cast("uchar", shift=False)

        return


def mask_eval_handler(result_mask, progress):
    """
    Function to display progress of pyvips operation.

    Inputs
        - Pyvips instance
        - Progress instance

    Outputs
        - Progress prompt in log
    """

    global savelog_mask

    percent = int(np.round(progress.percent))
    if percent % 10 == 0 and percent not in savelog_mask:
        handler_log.log(35, f" - progress {progress.percent}%")
        savelog_mask.append(percent)

    time.sleep(1)

    return


def image_eval_handler(result, progress):
    """
    Function to display progress of pyvips operation.

    Inputs
        - Pyvips instance
        - Progress instance

    Outputs
        - Progress prompt in log
    """

    global savelog_image

    percent = int(np.round(progress.percent))
    if percent % 10 == 0 and percent not in savelog_image:
        handler_log.log(35, f" - progress {progress.percent}%")
        savelog_image.append(percent)

    time.sleep(1)

    return


def perform_blending(result_image, result_mask, full_res_fragments, log, blend_dir):
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
    tilesize = 8192
    num_xtiles = int(np.ceil(width / tilesize))
    num_ytiles = int(np.ceil(height / tilesize))

    start = time.time()

    # Loop over columns
    for x in range(num_xtiles):

        log.log(35, f" - blending column {x+1}/{num_xtiles}")

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
                    image_patch = f.final_image.crop(
                        x * tilesize, y * tilesize, new_tilesize[0], new_tilesize[1],
                    )
                    image = np.ndarray(
                        buffer=image_patch.write_to_memory(),
                        dtype=np.uint8,
                        shape=[new_tilesize[1], new_tilesize[0], image_patch.bands],
                    )

                    mask_patch = f.fullres_mask_rot.crop(
                        x * tilesize, y * tilesize, new_tilesize[0], new_tilesize[1],
                    )
                    mask = np.ndarray(
                        buffer=mask_patch.write_to_memory(),
                        dtype=np.uint8,
                        shape=[new_tilesize[1], new_tilesize[0]],
                    )

                    images[f.name] = image
                    masks[f.name] = mask

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
                        f"{blend_dir}/row{str(y).zfill(4)}_col{str(x).zfill(4)}.png"
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

    comp_time = int((time.time() - start) / 60)

    return result_image, comp_time


def generate_full_res(parameters, log):
    """
    Function to generate the final full resolution stitching results.

    Inputs
        - Dictionary with parameters
        - Logging instance

    Outputs
        - Full resolution blended image
    """

    global handler_log, savelog_image, savelog_mask
    savelog_image, savelog_mask = [0], [0]
    handler_log = copy.deepcopy(log)

    names = parameters["fragment_names"]
    full_res_fragments = [FullResImage(parameters, name) for name in names]

    blend_dir = f"{parameters['results_dir']}/highres/blend_summary"

    log.log(parameters["my_level"], "Processing full resolution fragments")
    start = time.time()
    for f in full_res_fragments:
        log.log(parameters["my_level"], f" - fragment '{f.name}'")

        f.load_images()
        f.get_scaling()
        f.get_rotation()
        f.process_mask()
        f.process_image()

    log.log(
        parameters["my_level"], f" > finished in {int((time.time()-start)/60)} mins!\n"
    )

    # Add all images as the default stitching method
    result_image = pyvips.Image.sum([f.final_image for f in full_res_fragments])
    if not result_image.format == "uchar":
        result_image = result_image.cast("uchar", shift=False)

    # Do the same for masks
    result_mask = pyvips.Image.sum([f.fullres_mask_rot for f in full_res_fragments])
    if not result_mask.format == "uchar":
        result_mask = result_mask.cast("uchar", shift=False)

    """
    # Save full resolution mask if needed
    log.log(parameters["my_level"], "Saving full resolution mask")
    start = time.time()
    result_mask.set_progress(True)
    result_mask.signal_connect("eval", mask_eval_handler)
    result_mask.tiffsave(
        f"{parameters['results_dir']}/highres/fullres_mask.tif",
        tile=True,
        compression="lzw",
        bigtiff=True,
        pyramid=True,
        Q=80,
    )
    log.log(parameters["my_level"], f" > finished in {int((time.time()-start)/60)} mins!\n")
    """

    log.log(parameters["my_level"], "Blending areas of overlap")
    result, comp_time = perform_blending(
        result_image, result_mask, full_res_fragments, log, blend_dir
    )
    log.log(parameters["my_level"], f" > finished in {comp_time} mins!\n")

    log.log(parameters["my_level"], "Saving full resolution result")
    start = time.time()
    result.set_progress(True)
    result.signal_connect("eval", image_eval_handler)
    result.tiffsave(
        f"{parameters['results_dir']}/highres/fullres_image.tif",
        tile=True,
        compression="jpeg",
        bigtiff=True,
        pyramid=True,
        Q=80,
    )
    log.log(
        parameters["my_level"], f" > finished in {int((time.time()-start)/60)} mins!\n"
    )

    return
