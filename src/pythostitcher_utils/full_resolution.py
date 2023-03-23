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
from .gradient_blending import perform_blending

os.environ["VIPS_CONCURRENCY"] = "20"


class FullResImage:
    """
    Class for the full resolution fragments. This class contains several methods to
    process the full resolution fragments based on the transformation obtained by
    PythoStitcher.
    """

    def __init__(self, parameters, idx):

        self.idx = idx
        self.output_res = parameters["output_res"]
        self.raw_image_name = parameters["raw_image_names"][self.idx]
        self.raw_mask_name = parameters["raw_mask_names"][self.idx]
        self.save_dir = parameters["sol_save_dir"]
        self.data_dir = parameters["data_dir"]
        self.raw_image_path = self.data_dir.joinpath("raw_images", self.raw_image_name)
        self.orientation = parameters["detected_configuration"][self.raw_image_name].lower()
        self.rot_k = parameters["rot_steps"][self.raw_image_name]

        if self.raw_mask_name:
            self.raw_mask_path = parameters["data_dir"].joinpath("raw_masks", self.raw_mask_name)

        self.last_res = parameters["resolutions"][-1]
        self.res_name = get_resname(self.last_res)
        self.slice_idx = parameters["slice_idx"]
        self.ps_level = parameters["image_level"]

        self.data_dir = parameters["data_dir"]
        self.ps_mask_path = self.save_dir.joinpath(
            "images", self.slice_idx, self.res_name, f"fragment_{self.orientation}_mask.png"
        )

        return

    def load_images(self):
        """
        Load the raw image, raw mask and the PythoStitcher mask from the highest
        resolution.
        """

        # Get full res image, mask and pythostitcher mask
        self.opener = mir.MultiResolutionImageReader()
        self.raw_image = self.opener.open(str(self.raw_image_path))
        self.raw_mask = self.opener.open(str(self.raw_mask_path))

        self.ps_mask = cv2.imread(str(self.ps_mask_path))
        self.ps_mask = cv2.cvtColor(self.ps_mask, cv2.COLOR_BGR2GRAY)

        return

    def get_scaling(self):
        """
        Get scaling factors to go from PythoStitcher resolution to desired output resolution
        """

        # Get full resolution dims
        self.raw_image_dims = self.raw_image.getLevelDimensions(0)
        self.raw_mask_dims = self.raw_mask.getLevelDimensions(0)

        # Obtain the resolution (µm/pixel) for each level
        n_levels = self.raw_image.getNumberOfLevels()
        ds_per_level = [self.raw_image.getLevelDownsample(i) for i in range(n_levels)]
        res_per_level = [
            self.raw_image.getSpacing()[0] * scale
            for i, scale in zip(range(n_levels), ds_per_level)
        ]

        # Get the optimal level based on the desired output resolution
        self.output_level = np.argmin([(i - self.output_res) ** 2 for i in res_per_level])

        assert self.output_level <= self.ps_level, (
            f"Resolution level of the output image must be lower than the PythoStitcher "
            f"resolution level. Provided utput level is {self.output_level}, while "
            f"PythoStitcher level is {self.ps_level}. Please increase the output resolution."
        )

        # Get image on this optimal output level
        if self.raw_image_path.suffix == ".mrxs":
            self.outputres_image = pyvips.Image.new_from_file(
                str(self.raw_image_path), level=self.output_level
            )
        elif self.raw_image_path.suffix == ".tif":
            self.outputres_image = pyvips.Image.new_from_file(
                str(self.raw_image_path), page=self.output_level
            )
        else:
            raise ValueError("currently we only support mrxs and tif files")

        # Get new image dims
        self.outputres_image_dims = (
            self.outputres_image.get("width"),
            self.outputres_image.get("height"),
        )

        # Get scaling factor raw mask and final output resolution
        self.scaling_ps2outputres = 2 ** (self.ps_level - self.output_level) / self.last_res

        # Dimension of final stitching result
        self.target_dims = [int(i * self.scaling_ps2outputres) for i in self.ps_mask.shape]

        # Get the optimal transformation obtained with the genetic algorithm
        self.ps_tform = np.load(
            f"{self.save_dir}/tform/{self.res_name}_tform_final.npy", allow_pickle=True,
        ).item()

        # Upsample it to use it for the final image
        self.highres_tform = [
            int(self.ps_tform[self.orientation][0] * self.scaling_ps2outputres),
            int(self.ps_tform[self.orientation][1] * self.scaling_ps2outputres),
            np.round(self.ps_tform[self.orientation][2], 1),
            tuple([int(i * self.scaling_ps2outputres) for i in self.ps_tform[self.orientation][3]]),
            tuple([int(i * self.scaling_ps2outputres) for i in self.ps_tform[self.orientation][4]]),
        ]

        return

    def process_mask(self):
        """
        Process the mask to a full resolution version
        """

        # Get mask which is closest to 4k image. This is an empirical trade-off
        # between feasible image processing with opencv and mask resolution
        best_mask_output_dims = 4000
        all_mask_dims = [
            self.raw_mask.getLevelDimensions(i) for i in range(self.raw_mask.getNumberOfLevels())
        ]
        mask_ds_level = np.argmin([(i[0] - best_mask_output_dims) ** 2 for i in all_mask_dims])

        original_mask = self.raw_mask.getUCharPatch(
            startX=0,
            startY=0,
            width=int(all_mask_dims[mask_ds_level][0]),
            height=int(all_mask_dims[mask_ds_level][1]),
            level=int(mask_ds_level),
        )

        # Convert mask for opencv processing
        original_mask = original_mask / np.max(original_mask)
        original_mask = (original_mask * 255).astype("uint8")
        self.scaling_mask2outputres = self.outputres_image_dims[0] / original_mask.shape[1]

        # Get information on all connected components in the mask
        num_labels, original_mask, stats, _ = cv2.connectedComponentsWithStats(
            original_mask, connectivity=4
        )

        # Extract largest connected component
        largest_cc_label = np.argmax(stats[1:, -1]) + 1
        original_mask = ((original_mask == largest_cc_label) * 255).astype("uint8")

        # Some morphological operations for cleaning up edges
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(20, 20))
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
        floodfill_mask = np.zeros((original_mask.shape[0] + 2, original_mask.shape[1] + 2)).astype(
            "uint8"
        )
        _, _, original_mask, _ = cv2.floodFill(original_mask, floodfill_mask, seedpoint, 255)
        original_mask = (
            1 - original_mask[temp_pad + 1 : -(temp_pad + 1), temp_pad + 1 : -(temp_pad + 1)]
        )

        # Get nonzero indices and crop
        self.r_idx, self.c_idx = np.nonzero(original_mask)
        original_mask = original_mask[
            np.min(self.r_idx) : np.max(self.r_idx), np.min(self.c_idx) : np.max(self.c_idx),
        ]

        # Convert to pyvips array
        height, width = original_mask.shape
        bands = 1
        dformat = "uchar"
        self.outputres_mask = pyvips.Image.new_from_memory(
            original_mask.ravel(), width, height, bands, dformat
        )

        self.outputres_mask = self.outputres_mask.resize(self.scaling_mask2outputres)

        self.outputres_mask = self.outputres_mask.rotate(-self.rot_k * 90)

        # Pad image with zeros
        self.outputres_mask = self.outputres_mask.gravity(
            "centre", self.target_dims[1], self.target_dims[0]
        )

        return

    def process_image(self):
        """
        Process the raw image to the full resolution version
        """

        # Dispose of alpha channel if applicable
        if self.outputres_image.hasalpha():
            self.outputres_image = self.outputres_image.flatten()

        # Get cropping indices
        rmin, rmax = (
            int(self.scaling_mask2outputres * np.min(self.r_idx)),
            int(self.scaling_mask2outputres * np.max(self.r_idx)),
        )
        cmin, cmax = (
            int(self.scaling_mask2outputres * np.min(self.c_idx)),
            int(self.scaling_mask2outputres * np.max(self.c_idx)),
        )
        width = cmax - cmin
        height = rmax - rmin

        # Crop image
        self.outputres_image = self.outputres_image.crop(cmin, rmin, width, height)

        # Rotate image
        self.outputres_image = self.outputres_image.rotate(-self.rot_k * 90)

        # Pad image with zeros
        self.outputres_image = self.outputres_image.gravity(
            "centre", self.target_dims[1], self.target_dims[0]
        )

        # Get transformation matrix
        rotmat = cv2.getRotationMatrix2D(
            center=self.highres_tform[3], angle=self.highres_tform[2], scale=1
        )
        rotmat[0, 2] += self.highres_tform[0]
        rotmat[1, 2] += self.highres_tform[1]

        # Apply affine transformation
        self.outputres_image = self.outputres_image.affine(
            (rotmat[0, 0], rotmat[0, 1], rotmat[1, 0], rotmat[1, 1]),
            interpolate=pyvips.Interpolate.new("nearest"),
            odx=rotmat[0, 2],
            ody=rotmat[1, 2],
            oarea=[0, 0, self.highres_tform[4][1], self.highres_tform[4][0]],
        )

        self.outputres_mask = self.outputres_mask.affine(
            (rotmat[0, 0], rotmat[0, 1], rotmat[1, 0], rotmat[1, 1]),
            interpolate=pyvips.Interpolate.new("nearest"),
            odx=rotmat[0, 2],
            ody=rotmat[1, 2],
            oarea=[0, 0, self.highres_tform[4][1], self.highres_tform[4][0]],
        )

        if not self.outputres_image.format == "uchar":
            self.outputres_image = self.outputres_image.cast("uchar", shift=False)

        if not self.outputres_mask.format == "uchar":
            self.outputres_mask = self.outputres_mask.cast("uchar", shift=False)

        # Apply mask to images
        self.final_image = self.outputres_image.multiply(self.outputres_mask)
        if not self.final_image.format == "uchar":
            self.final_image = self.final_image.cast("uchar", shift=False)

        return


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

    parameters["blend_dir"] = parameters["sol_save_dir"].joinpath("highres", "blend_summary")

    # Initiate class for each fragment to handle full resolution image
    full_res_fragments = [FullResImage(parameters, idx) for idx in range(parameters["n_fragments"])]

    log.log(parameters["my_level"], "Processing high resolution fragments")
    start = time.time()
    for f in full_res_fragments:
        log.log(parameters["my_level"], f" - '{f.raw_image_path.name}'")

        # Transform each fragment such that all final images can just be added as an
        # initial stitch
        f.load_images()
        f.get_scaling()
        f.process_mask()
        f.process_image()

    log.log(
        parameters["my_level"], f" > finished in {int(np.ceil((time.time()-start)/60))} mins!\n"
    )

    # Add all images as the default stitching method
    result_image = pyvips.Image.sum([f.final_image for f in full_res_fragments])
    if not result_image.format == "uchar":
        result_image = result_image.cast("uchar", shift=False)

    # Do the same for masks
    result_mask = pyvips.Image.sum([f.outputres_mask for f in full_res_fragments])
    if not result_mask.format == "uchar":
        result_mask = result_mask.cast("uchar", shift=False)

    # Save temp .tif version of mask for later use in blending
    parameters["tif_mask_path"] = str(
        parameters["sol_save_dir"].joinpath("highres", "temp_mask.tif")
    )
    log.log(parameters["my_level"], f"Saving temporary mask at {parameters['output_res']} µm/pixel")
    start = time.time()
    result_mask.write_to_file(
        parameters["tif_mask_path"], tile=True, compression="lzw", bigtiff=True, pyramid=True,
    )
    log.log(
        parameters["my_level"], f" > finished in {int(np.ceil((time.time()-start)/60))} mins!\n"
    )

    # Perform blending in areas of overlap
    log.log(parameters["my_level"], "Blending areas of overlap")
    result_image, comp_time = perform_blending(
        result_image, result_mask, full_res_fragments, log, parameters
    )
    log.log(parameters["my_level"], f" > finished in {comp_time} mins!\n")

    # Save final result
    log.log(
        parameters["my_level"], f"Saving blended end result at {parameters['output_res']} µm/pixel"
    )
    start = time.time()
    result_image.write_to_file(
        str(
            parameters["sol_save_dir"].joinpath(
                "highres", f"stitched_image_{parameters['output_res']}_micron.tif"
            )
        ),
        tile=True,
        compression="jpeg",
        bigtiff=True,
        pyramid=True,
        Q=20,
    )
    log.log(
        parameters["my_level"], f" > finished in {int(np.ceil((time.time()-start)/60))} mins!\n"
    )

    return
