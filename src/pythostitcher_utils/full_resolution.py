import numpy as np
import os
import pyvips
import cv2
import multiresolutionimageinterface as mir
import time
import copy
import matplotlib.pyplot as plt
import logging
import math

from .get_resname import get_resname
from .fuse_images_highres import fuse_images_highres
from .gradient_blending import perform_blending
from .line_utils import apply_im_tform_to_coords
from .landmark_evaluation import evaluate_landmarks

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

        self.coord_path = parameters["save_dir"].joinpath(
            "landmarks", f"fragment{self.idx + 1}_coordinates.npy"
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

        # Load landmark points for quantification of residual registration error
        self.val_coordinates = np.load(self.coord_path, allow_pickle=True).item()
        self.line_a = self.val_coordinates["a"]
        self.line_b = self.val_coordinates["b"]

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
        self.output_level = int(np.argmin([(i - self.output_res) ** 2 for i in res_per_level]))
        self.output_spacing = self.raw_image.getSpacing()[0] * self.raw_image.getLevelDownsample(
            self.output_level)

        assert self.output_level <= self.ps_level, (
            f"Resolution level of the output image must be lower than the PythoStitcher "
            f"resolution level. Provided output level is {self.output_level}, while "
            f"PythoStitcher level is {self.ps_level}. Please increase the output resolution."
        )

        # Get image on this optimal output level
        if self.raw_image_path.suffix == ".mrxs":
            self.outputres_image = pyvips.Image.new_from_file(
                str(self.raw_image_path), level=self.output_level
            )
        elif self.raw_image_path.suffix in [".tif", ".tiff"]:
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

        # Get scaling factor raw mask and coords wrt to final output resolution
        self.scaling_ps2outputres = 2 ** (self.ps_level - self.output_level) / self.last_res
        self.scaling_coords2outputres = self.raw_image_dims[0] / self.outputres_image_dims[0]

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

    def get_tissue_seg_mask(self):
        """
        Get the mask from the tissue segmentation algorithm and postprocess.
        """

        # Get mask which is closest to 2k image. This is an empirical trade-off
        # between feasible image processing with opencv and mask resolution
        best_mask_output_dims = 2000
        all_mask_dims = [
            self.raw_mask.getLevelDimensions(i) for i in range(self.raw_mask.getNumberOfLevels())
        ]
        self.mask_ds_level = np.argmin([(i[0] - best_mask_output_dims) ** 2 for i in all_mask_dims])

        self.tissueseg_mask = self.raw_mask.getUCharPatch(
            startX=0,
            startY=0,
            width=int(all_mask_dims[self.mask_ds_level][0]),
            height=int(all_mask_dims[self.mask_ds_level][1]),
            level=int(self.mask_ds_level),
        )

        # Convert mask for opencv processing
        self.tissueseg_mask = self.tissueseg_mask / np.max(self.tissueseg_mask)
        self.tissueseg_mask = (self.tissueseg_mask * 255).astype("uint8")
        self.scaling_mask2outputres = self.outputres_image_dims[0] / self.tissueseg_mask.shape[1]

        # Get information on all connected components in the mask
        num_labels, self.tissueseg_mask, stats, _ = cv2.connectedComponentsWithStats(
            self.tissueseg_mask, connectivity=8
        )

        # Extract largest connected component
        largest_cc_label = np.argmax(stats[1:, -1]) + 1
        self.tissueseg_mask = ((self.tissueseg_mask == largest_cc_label) * 255).astype("uint8")

        # Some morphological operations for cleaning up edges
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(20, 20))
        self.tissueseg_mask = cv2.morphologyEx(
            src=self.tissueseg_mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=2
        )

        # Slightly enlarge temporary
        temp_pad = int(0.05 * self.tissueseg_mask.shape[0])
        self.tissueseg_mask = np.pad(
            self.tissueseg_mask,
            [[temp_pad, temp_pad], [temp_pad, temp_pad]],
            mode="constant",
            constant_values=0,
        )

        # Flood fill to remove holes inside mask
        seedpoint = (0, 0)
        floodfill_mask = np.zeros((self.tissueseg_mask.shape[0] + 2, self.tissueseg_mask.shape[1] +
                                   2)).astype(
            "uint8"
        )
        _, _, self.tissueseg_mask, _ = cv2.floodFill(self.tissueseg_mask, floodfill_mask,
                                                     seedpoint,
                                                   255)
        self.tissueseg_mask = self.tissueseg_mask[
            temp_pad + 1: -(temp_pad + 1), temp_pad + 1: -(temp_pad + 1)
        ]
        self.tissueseg_mask = 1 - self.tissueseg_mask

        return

    def get_otsu_mask(self):
        """
        Get mask based on Otsu thresholding.
        """

        # Get image of same size as tissue segmentation mask

        ### EXPERIMENTAL --- get the image level based on the mask level ###
        # New code
        im_vs_mask_ds = self.raw_image.getLevelDimensions(0)[0]/self.raw_mask.getLevelDimensions(
            0)[0]
        image_ds_level = int(self.mask_ds_level +
                             int(math.log2(im_vs_mask_ds)))

        # Previous code
        # image_ds_level = int(self.mask_ds_level + int(math.log2(self.scaling_coords2outputres)))
        ### \\\ EXPERIMENTAL ###

        image_ds_dims = self.raw_image.getLevelDimensions(image_ds_level)
        self.otsu_image = self.raw_image.getUCharPatch(
            startX=0,
            startY=0,
            width=int(image_ds_dims[0]),
            height=int(image_ds_dims[1]),
            level=int(image_ds_level),
        )

        ### EXPERIMENTAL - shift black background to white for correct otsu thresholding###
        self.otsu_image[np.all(self.otsu_image<10, axis=2)] = 255
        ### \\\ EXPERIMENTAL ###

        image_hsv = cv2.cvtColor(self.otsu_image, cv2.COLOR_RGB2HSV)
        image_hsv = cv2.medianBlur(image_hsv[:, :, 1], 7)
        _, self.otsu_mask = cv2.threshold(image_hsv, 0, 255, cv2.THRESH_OTSU +
                                          cv2.THRESH_BINARY)
        self.otsu_mask = (self.otsu_mask / np.max(self.otsu_mask)).astype("uint8")

        # Postprocess the mask a bit
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15, 15))
        self.otsu_mask = cv2.morphologyEx(
            src=self.otsu_mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=3
        )

        return


    def combine_masks(self):
        """
        Combine Otsu mask and tissue segmentation mask, similar in preprocessing scripts.
        """

        # First process the coordinates
        self.line_a = (self.line_a / self.scaling_coords2outputres).astype("int")
        self.line_b = (self.line_b / self.scaling_coords2outputres).astype("int")

        # Combine masks
        self.final_mask = self.otsu_mask * self.tissueseg_mask

        # Postprocess similar to tissue segmentation mask. Get largest cc and floodfill.
        num_labels, labeled_im, stats, _ = cv2.connectedComponentsWithStats(
            self.final_mask, connectivity=8
        )
        assert num_labels > 1, "mask is empty"
        largest_cc_label = np.argmax(stats[1:, -1]) + 1
        self.final_mask = ((labeled_im == largest_cc_label) * 255).astype("uint8")

        # Flood fill
        offset = 5
        self.final_mask = np.pad(
            self.final_mask,
            [[offset, offset], [offset, offset]],
            mode="constant",
            constant_values=0
        )

        seedpoint = (0, 0)
        floodfill_mask = np.zeros(
            (self.final_mask.shape[0] + 2, self.final_mask.shape[1] + 2)).astype("uint8")
        _, _, self.final_mask, _ = cv2.floodFill(self.final_mask, floodfill_mask, seedpoint, 255)
        self.final_mask = self.final_mask[1 + offset:-1 - offset, 1 + offset:-1 - offset]
        self.final_mask = 1 - self.final_mask

        # Crop to nonzero pixels for efficient saving
        self.r_idx, self.c_idx = np.nonzero(self.final_mask)
        self.final_mask = self.final_mask[np.min(self.r_idx): np.max(self.r_idx),
                          np.min(self.c_idx): np.max(self.c_idx)]

        # Convert to pyvips array
        height, width = self.final_mask.shape
        bands = 1
        dformat = "uchar"
        self.outputres_mask = pyvips.Image.new_from_memory(
            self.final_mask.ravel(), width, height, bands, dformat
        )

        self.outputres_mask = self.outputres_mask.resize(self.scaling_mask2outputres)

        if self.rot_k in range(1, 4):
            self.outputres_mask = self.outputres_mask.rotate(self.rot_k * 90)

        # Pad image with zeros
        self.outputres_mask = self.outputres_mask.gravity(
            "centre", self.target_dims[1], self.target_dims[0]
        )

        return

    def process_mask(self):
        """
        Process the mask to a full resolution version
        """

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

        if self.rot_k in range(1, 4):
            self.outputres_mask = self.outputres_mask.rotate(self.rot_k * 90)

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

        # Prevent bad crop area errors due to upsampling/rounding errors
        width = np.min([cmax, self.outputres_image.width]) - cmin
        height = np.min([rmax, self.outputres_image.height]) - rmin

        # Crop image
        self.outputres_image = self.outputres_image.crop(cmin, rmin, width, height)

        # Also apply cropping and rotation to landmark points
        self.line_a = np.vstack([self.line_a[:, 0] - cmin, self.line_a[:, 1] - rmin]).T
        self.line_b = np.vstack([self.line_b[:, 0] - cmin, self.line_b[:, 1] - rmin]).T

        # Apply rotation
        self.line_a = apply_im_tform_to_coords(
            self.line_a,
            self.outputres_image,
            self.rot_k
        )
        self.line_b = apply_im_tform_to_coords(
            self.line_b,
            self.outputres_image,
            self.rot_k
        )

        # Rotate image if necessary
        if self.rot_k in range(1, 4):
            self.outputres_image = self.outputres_image.rotate(self.rot_k * 90)

        # Pad image with zeros
        xpad = int((self.target_dims[1] - self.outputres_image.width)/2)
        ypad = int((self.target_dims[0] - self.outputres_image.height)/2)

        self.outputres_image = self.outputres_image.gravity(
            "centre", self.target_dims[1], self.target_dims[0]
        )

        # Also apply to landmark points
        self.line_a = np.vstack([self.line_a[:, 0] + xpad, self.line_a[:, 1] + ypad]).T
        self.line_b = np.vstack([self.line_b[:, 0] + xpad, self.line_b[:, 1] + ypad]).T

        # Get transformation matrix
        rotmat = cv2.getRotationMatrix2D(
            center=self.highres_tform[3], angle=self.highres_tform[2], scale=1
        )
        rotmat[0, 2] += self.highres_tform[0]
        rotmat[1, 2] += self.highres_tform[1]

        # Apply affine transformation to images
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

        # Apply affine transformation to coordinates
        ones = np.ones((len(self.line_a), 1))

        self.line_a = np.hstack([self.line_a, ones]) @ rotmat.T
        self.line_b = np.hstack([self.line_b, ones]) @ rotmat.T

        self.line_a = self.line_a.astype("int")
        self.line_b = self.line_b.astype("int")

        if not self.outputres_image.format == "uchar":
            self.outputres_image = self.outputres_image.cast("uchar", shift=False)

        if not self.outputres_mask.format == "uchar":
            self.outputres_mask = self.outputres_mask.cast("uchar", shift=False)

        # Apply mask to images
        self.final_image = self.outputres_image.multiply(self.outputres_mask)
        if not self.final_image.format == "uchar":
            self.final_image = self.final_image.cast("uchar", shift=False)

        self.eval_dir = self.save_dir.joinpath("highres", "eval")
        if not self.eval_dir.is_dir():
            self.eval_dir.mkdir()

        rot_lines = {"a" : self.line_a, "b" : self.line_b}
        np.save(
            str(self.eval_dir.joinpath(f"fragment{self.idx+1}_coordinates.npy")),
            rot_lines
        )

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
        # f.process_mask()
        f.get_tissue_seg_mask()
        f.get_otsu_mask()
        f.combine_masks()
        f.process_image()

    ### DEBUGGING ###
    log.setLevel(logging.ERROR)

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
        parameters["tif_mask_path"],
        tile=True,
        compression="lzw",
        bigtiff=True,
        pyramid=True,
    )
    log.log(
        parameters["my_level"], f" > finished in {int(np.ceil((time.time()-start)/60))} mins!\n"
    )

    # Perform blending in areas of overlap
    log.log(parameters["my_level"], "Blending areas of overlap")
    start = time.time()
    result_image = perform_blending(
        result_image, result_mask, full_res_fragments, log, parameters
    )
    log.log(
        parameters["my_level"], f" > finished in {int(np.ceil((time.time()-start)/60))} mins!\n"
    )

    # Remove temporary mask
    parameters["sol_save_dir"].joinpath("highres", "temp_mask.tif").unlink()

    # Ensure output image has spacing attribute
    xyres = 1000 / full_res_fragments[0].output_spacing
    result_image_save = result_image.copy(xres=xyres, yres=xyres)

    # Save final result
    log.log(
        parameters["my_level"], f"Saving blended end result at {parameters['output_res']} µm/pixel"
    )
    start = time.time()

    result_image_save.write_to_file(
        str(
            parameters["sol_save_dir"].joinpath(
                "highres", f"stitched_image_{parameters['output_res']}_micron.tif"
            )
        ),
        tile=True,
        compression="jpeg",
        bigtiff=True,
        pyramid=True,
        Q=80,
    )
    log.log(
        parameters["my_level"], f" > finished in {int(np.ceil((time.time()-start)/60))} mins!\n"
    )

    # Evaluate residual registration mismatch
    evaluate_landmarks(parameters)

    # Clean up for next solution
    del full_res_fragments, result_mask, result_image, result_image_save

    return
