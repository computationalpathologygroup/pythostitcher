import pickle
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import TheilSenRegressor

from .get_resname import get_resname
from .transformations import warp_2d_points, warp_image


class Fragment:
    """
    Class for the individual fragments. This class mainly consists of methods regarding
    image processing and holds several attributes that are required for these processing
    steps.
    """

    def __init__(self, im_path, fragment_name, kwargs):
        self.final_orientation = fragment_name.lower()
        self.original_name = im_path
        self.iteration = kwargs["iteration"]
        self.patient_idx = kwargs["patient_idx"]
        self.slice_idx = kwargs["slice_idx"]
        self.num_fragments = kwargs["n_fragments"]
        self.save_dir = kwargs["save_dir"]
        self.sol_save_dir = kwargs["sol_save_dir"]

        self.resolutions = kwargs["resolutions"]
        self.nbins = kwargs["nbins"]
        self.hist_sizes = kwargs["hist_sizes"]
        self.data_dir = kwargs["data_dir"]
        self.original_image_idx = kwargs["fragment_names"].index(self.final_orientation) + 1
        self.im_path = self.save_dir.joinpath(
            "preprocessed_images", f"fragment{self.original_image_idx}.png"
        )
        self.mask_path = self.save_dir.joinpath(
            "preprocessed_masks", f"fragment{self.original_image_idx}.png"
        )
        self.res = self.resolutions[self.iteration]
        self.res_name = get_resname(self.res)
        self.pad_fraction = kwargs["pad_fraction"]

        if self.num_fragments == 2:
            self.rot_k = kwargs["rot_steps"][self.original_name]

        self.complementary_fragments_pair = {
            "ul": "ur",
            "ur": "ul",
            "ll": "lr",
            "lr": "ll",
            "left": "right",
            "right": "left",
            "top": "bottom",
            "bottom": "top",
        }
        self.complementary_fragments_total = {"ul": "ll", "ur": "ll", "ll": "ul", "lr": "ul"}

        if self.num_fragments == 4:
            self.stitch_edge_dict = self.save_dir.joinpath(
                "configuration_detection", "stitch_edges.txt"
            )
            assert (
                self.stitch_edge_dict.exists()
            ), "received invalid path to 'stitch_edges.txt' file"

        return

    def read_transforms(self):
        """
        Method to read the transformation info for each fragment.
        """

        # Get the fragment classification which was performed previously. Note that this
        # label corresponds to the inner point of the fragment and not the final position
        # in the reconstruction, i.e. label "UL" indicates that the inner point of said
        # fragment is located at the upper left corner and NOT that the fragment should
        # be located in the upper left quadrant of the final reconstruction. Note that
        # these are directly diagonally opposite.
        stitch_label_dict = dict()
        with open(self.stitch_edge_dict, "r") as f:
            lines = f.readlines()
            lines = [i.rstrip("\n") for i in lines]
            for line in lines:
                key, value = line.split(":")
                stitch_label_dict[key] = value

        # Determine the initial orientation of the fragment.
        self.all_labels_loop = ["ur", "lr", "ll", "ul"] * 2
        self.classifier_label = stitch_label_dict[
            f"fragment{self.original_image_idx}.png"
        ].lower()
        self.init_orientation = self.all_labels_loop[
            self.all_labels_loop.index(self.classifier_label) + 2
        ]

        # Determine how many times the fragment needs to be located ccw to obtain its
        # final position
        idx_post = self.all_labels_loop.index(self.final_orientation)
        idx_pre = self.all_labels_loop.index(self.init_orientation)
        if idx_post > idx_pre:
            self.rot_k = idx_post - idx_pre
        else:
            self.rot_k = idx_post + 4 - idx_pre

        return

    def read_image(self):
        """
        Method to read the fragment image. Input images should preferably be .tiff or
        .tif files, but can probably be any format as long as it is supported by opencv.
        """

        self.original_image = cv2.imread(str(self.im_path))
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # Apply rotation/flipping
        if self.rot_k != 0:
            self.original_image = np.rot90(self.original_image, k=self.rot_k, axes=(1, 0))

        return

    def downsample_image(self):
        """
        Method to downsample original image.
        """

        # Get target size for downsampled image
        self.target_size = (
            int(np.round(self.res * np.shape(self.original_image)[1])),
            int(np.round(self.res * np.shape(self.original_image)[0])),
        )

        # Get downsampled gray image
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        self.gray_image = cv2.resize(self.gray_image, self.target_size)

        # Get downsampled regular image
        self.colour_image = cv2.resize(self.original_image, self.target_size)

        return

    def segment_tissue(self):
        """
        Method to obtain tissue segmentation mask at a given resolution level. This just
        loads the mask that was previously obtained by the background segmentation
        algorithm. A later version of Pythostitcher could perhaps integrate this
        segmentation model.
        """

        # Directly get mask from original image
        self.mask = cv2.imread(str(self.mask_path))
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_RGB2GRAY)

        # Rotate and resize mask to match image
        if self.rot_k != 0:
            self.mask = np.rot90(self.mask, k=self.rot_k, axes=(1, 0))
        self.mask = (cv2.resize(self.mask, self.target_size) > 128) * 1

        # Apply mask to image
        self.colour_image = (self.colour_image * self.mask[:, :, np.newaxis]).astype("uint8")
        self.gray_image = (self.gray_image * self.mask).astype("uint8")

        return

    def apply_masks(self):
        """
        Method to apply the previously obtained tissue mask to the images.
        """

        # Crop the image
        c, r = np.nonzero(self.mask)
        cmin, cmax = np.min(c), np.max(c)
        rmin, rmax = np.min(r), np.max(r)

        self.colour_image = self.colour_image[cmin:cmax, rmin : rmax + 1]
        self.gray_image = self.gray_image[cmin:cmax, rmin : rmax + 1]
        self.mask = self.mask[cmin:cmax, rmin : rmax + 1]

        # For the lowest resolution we apply a fixed padding rate. For higher resolutions
        # we compute the ideal padding based on the previous resolution such that the
        # fragment retains the same image/pad ratio as the previous image.
        if self.iteration == 0:
            self.initial_pad = int(np.max(self.target_size) * self.pad_fraction)
            self.colour_image = np.pad(
                self.colour_image,
                [
                    [self.initial_pad, self.initial_pad],
                    [self.initial_pad, self.initial_pad],
                    [0, 0],
                ],
            )
            self.gray_image = np.pad(
                self.gray_image,
                [[self.initial_pad, self.initial_pad], [self.initial_pad, self.initial_pad],],
            )
            self.mask = np.pad(
                self.mask,
                [[self.initial_pad, self.initial_pad], [self.initial_pad, self.initial_pad],],
            )
        else:
            prev_res_image_path = (
                f"{self.sol_save_dir}/images/{self.slice_idx}/"
                f"{get_resname(self.resolutions[self.iteration-1])}/"
                f"fragment_{self.final_orientation}_gray.png"
            )
            prev_res_image = cv2.imread(prev_res_image_path)
            prev_res_image_shape = np.shape(prev_res_image)
            ratio = self.res / self.resolutions[self.iteration - 1]
            current_res_image_shape = np.shape(self.gray_image)
            ideal_current_res_image_shape = [ratio * i for i in prev_res_image_shape]
            pad = [
                int((a - b) / 2)
                for a, b in zip(ideal_current_res_image_shape, current_res_image_shape)
            ]

            self.colour_image = np.pad(
                self.colour_image, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
            )
            self.gray_image = np.pad(self.gray_image, [[pad[0], pad[0]], [pad[1], pad[1]]])
            self.mask = np.pad(self.mask, [[pad[0], pad[0]], [pad[1], pad[1]]])

        return

    def save_fragment(self):
        """
        Method to save the current fragment images and the class itself with all its
        parameters.
        """

        self.fragment_savepath = (
            f"{self.sol_save_dir}/images/{self.slice_idx}/"
            f"{self.res_name}/fragment_{self.final_orientation}"
        )

        # Save mask, grayscale and colour image. Remove these from the class after saving
        # to prevent double saving.
        cv2.imwrite(
            self.fragment_savepath + "_mask.png",
            cv2.cvtColor((self.mask * 255).astype("uint8"), cv2.COLOR_GRAY2BGR),
        )

        cv2.imwrite(
            self.fragment_savepath + "_gray.png",
            cv2.cvtColor(self.gray_image.astype("uint8"), cv2.COLOR_GRAY2BGR),
        )

        cv2.imwrite(
            self.fragment_savepath + "_colour.png",
            cv2.cvtColor(self.colour_image.astype("uint8"), cv2.COLOR_RGB2BGR),
        )

        del self.mask, self.gray_image, self.colour_image, self.original_image

        # Save the fragment info without the images
        with open(self.fragment_savepath, "wb") as savefile:
            pickle.dump(self, savefile)

        return

    def load_images(self):
        """
        Method to load previously preprocessed fragment images.
        """

        # Read all relevant images. Take into account that opencv reads images in BGR
        # rather than RGB.
        basepath_load = f"{self.sol_save_dir}/images/{self.slice_idx}/{self.res_name}"
        self.mask = cv2.imread(f"{basepath_load}/fragment_{self.final_orientation}_mask.png")
        if len(self.mask.shape) == 3:
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        self.gray_image = cv2.imread(f"{basepath_load}/fragment_{self.final_orientation}_gray.png")
        self.gray_image = cv2.cvtColor(self.gray_image, cv2.COLOR_BGR2GRAY)
        self.colour_image = cv2.imread(
            f"{basepath_load}/fragment_{self.final_orientation}_colour.png"
        )
        self.colour_image = cv2.cvtColor(self.colour_image, cv2.COLOR_BGR2RGB)

        # Make a copy of all images
        self.mask_original = copy.copy(self.mask)
        self.gray_image_original = copy.copy(self.gray_image)
        self.colour_image_original = copy.copy(self.colour_image)

        return

    def get_bbox_corners(self, image):
        """
        Custom method to obtain bounding box corner coordinates of the smallest bounding
        box around the fragment. The way the corners are named is dependent on whether
        we are dealing with 2 or 4 tissue fragments. In the case of 2 fragments we only
        take the two corners which are closest to the stitching edge. In the case of 4
        fragments, we identify the inner point and name the other corners clockwise.
        """

        # Convert to uint8 for opencv processing
        image = (image / np.max(image) * 255).astype(np.uint8)

        # Obtain contour from mask
        self.cnt, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        self.cnt = np.squeeze(max(self.cnt, key=cv2.contourArea))
        self.cnt = self.cnt[::-1]

        # Convert bbox object to corner points. These corner points are always oriented
        # counter clockwise.
        self.bbox = cv2.minAreaRect(self.cnt)
        self.bbox_corners = cv2.boxPoints(self.bbox)

        # Get list of x-y values of contour
        x_points = [i[0] for i in self.cnt]
        y_points = [i[1] for i in self.cnt]
        distances = []
        mask_corners = []
        mask_corners_idxs = []

        ### We now basically have 2 approaches. If we have 2 fragments, we can just take
        ### the only 2 relevant box corners (the side to be stitched). In case of 4
        ### fragments, we have to make use of the property that the fragments are likely
        ### shaped as a quarter circle and we can name them according to their position
        ### in this quarter circle. In this case, corner A is the centerpoint and all
        ### other corners are named in clockwise direction.

        # Get distance from each corner to the mask
        for corner in self.bbox_corners:
            dist_x = [np.abs(corner[0] - x_point) for x_point in x_points]
            dist_y = [np.abs(corner[1] - y_point) for y_point in y_points]
            dist = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(dist_x, dist_y)]
            mask_idx = np.argmin(dist)
            mask_corners.append(self.cnt[mask_idx])
            mask_corners_idxs.append(mask_idx)
            distances.append(np.min(dist))

        mask_corners = np.array(mask_corners)

        # Case of 2 fragments
        if self.num_fragments == 2:

            # Define stitching edge based on two points with largest/smallest x/y
            # coordinates.
            if self.final_orientation == "top":
                corner_idxs = np.argsort(self.bbox_corners[:, 1])[-2:]
            elif self.final_orientation == "bottom":
                corner_idxs = np.argsort(self.bbox_corners[:, 1])[:2]
            elif self.final_orientation == "left":
                corner_idxs = np.argsort(self.bbox_corners[:, 0])[-2:]
            elif self.final_orientation == "right":
                corner_idxs = np.argsort(self.bbox_corners[:, 0])[:2]

            self.bbox_corner_a = self.bbox_corners[corner_idxs[0]]
            self.mask_corner_a = mask_corners[corner_idxs[0]]
            self.mask_corner_a_idx = mask_corners_idxs[corner_idxs[0]]
            self.bbox_corner_b = self.bbox_corners[corner_idxs[1]]
            self.mask_corner_b = mask_corners[corner_idxs[1]]
            self.mask_corner_b_idx = mask_corners_idxs[corner_idxs[1]]

        # Case of 4 fragments
        elif self.num_fragments == 4:

            center = np.mean(mask_corners, axis=0)
            corner_idxs = [0, 1, 2, 3] * 2

            if self.final_orientation == "ul":
                corner_a_idx = np.argmax(
                    (mask_corners[:, 0] > center[0]) * (mask_corners[:, 1] > center[1])
                )
            elif self.final_orientation == "ur":
                corner_a_idx = np.argmax(
                    (mask_corners[:, 0] < center[0]) * (mask_corners[:, 1] > center[1])
                )
            elif self.final_orientation == "ll":
                corner_a_idx = np.argmax(
                    (mask_corners[:, 0] > center[0]) * (mask_corners[:, 1] < center[1])
                )
            elif self.final_orientation == "lr":
                corner_a_idx = np.argmax(
                    (mask_corners[:, 0] < center[0]) * (mask_corners[:, 1] < center[1])
                )

            # Corner a is the inner corner
            self.bbox_corner_a = self.bbox_corners[corner_a_idx]
            self.mask_corner_a = mask_corners[corner_a_idx]
            self.mask_corner_a_idx = mask_corners_idxs[corner_a_idx]

            # All other corners are named clockwise
            corner_b_idx = corner_idxs[corner_a_idx + 1]
            self.bbox_corner_b = self.bbox_corners[corner_b_idx]
            self.mask_corner_b = mask_corners[corner_b_idx]
            self.mask_corner_b_idx = mask_corners_idxs[corner_b_idx]

            corner_c_idx = corner_idxs[corner_a_idx + 2]
            self.bbox_corner_c = self.bbox_corners[corner_c_idx]
            self.mask_corner_c = mask_corners[corner_c_idx]
            self.mask_corner_c_idx = mask_corners_idxs[corner_c_idx]

            corner_d_idx = corner_idxs[corner_a_idx + 3]
            self.bbox_corner_d = self.bbox_corners[corner_d_idx]
            self.mask_corner_d = mask_corners[corner_d_idx]
            self.mask_corner_d_idx = mask_corners_idxs[corner_d_idx]

        return

    def get_initial_transform(self):
        """
        Custom method to get the initial transformation consisting of rotation and
        cropping.
        """

        # Preprocess the rotation angle
        self.angle = self.bbox[2]
        if self.angle > 45:
            self.angle = self.angle - 90

        self.angle = np.round(self.angle, 1)

        # Apply rotation first
        rot_mat = cv2.getRotationMatrix2D(center=self.image_center_pre, angle=self.angle, scale=1)
        self.tform_image = cv2.warpAffine(
            src=self.gray_image, M=rot_mat, dsize=self.gray_image.shape[::-1]
        )
        self.rot_mask = cv2.warpAffine(
            src=self.mask_original, M=rot_mat, dsize=self.mask_original.shape[::-1]
        )

        # Get cropping parameters
        r, c = np.nonzero(self.tform_image)
        cmin, cmax = np.min(c), np.max(c)
        rmin, rmax = np.min(r), np.max(r)

        # Apply cropping parameters
        self.tform_image = self.tform_image[rmin:rmax, cmin:cmax]
        self.rot_mask = self.rot_mask[rmin:rmax, cmin:cmax]

        # Save cropping parameters as part of the transformation
        self.crop_trans_x = -cmin
        self.crop_trans_y = -rmin

        self.small_pad = int(self.initial_pad * 0.5)

        # Make dict with all possible padding outcomes to prevent a ton of if/else
        # statements for each case
        fragment_names = ["ul", "ur", "ll", "lr", "left", "right", "top", "bottom"]
        paddings = [
            [[self.small_pad, 0], [self.small_pad, 0]],
            [[self.small_pad, 0], [0, self.small_pad]],
            [[0, self.small_pad], [self.small_pad, 0]],
            [[0, self.small_pad], [0, self.small_pad]],
            [[self.small_pad, self.small_pad], [self.small_pad, 0]],
            [[self.small_pad, self.small_pad], [0, self.small_pad]],
            [[self.small_pad, 0], [self.small_pad, self.small_pad]],
            [[0, self.small_pad], [self.small_pad, self.small_pad]],
        ]
        padded_x = [
            self.small_pad,
            0,
            self.small_pad,
            0,
            self.small_pad,
            0,
            self.small_pad,
            self.small_pad,
        ]
        padded_y = [
            self.small_pad,
            self.small_pad,
            0,
            0,
            self.small_pad,
            self.small_pad,
            self.small_pad,
            0,
        ]

        pad_dict = dict()
        for fragment, pad, pad_x, pad_y in zip(fragment_names, paddings, padded_x, padded_y):
            pad_dict[fragment] = [pad, pad_x, pad_y]

        pad, pad_x, pad_y = pad_dict[self.final_orientation]
        self.tform_image = np.pad(self.tform_image, pad)
        self.pad_trans_x = pad_x
        self.pad_trans_y = pad_y

        return

    def get_tformed_images_pair(self, fragments):
        """
        Method to compute the transform necessary to align two horizontal pieces.
        This is basically a horizontal concatenation where both fragments are padded
        such that they will have the same shape.
        """

        # Get the complementary fragment
        other_fragment_name = self.complementary_fragments_pair[self.final_orientation]
        other_fragment = next(
            (i for i in fragments if i.final_orientation == other_fragment_name), None
        )

        # Epsilon value is necessary to ensure no overlap between images due to rounding
        # errors.
        eps = 1

        if self.final_orientation in ["ul", "left"]:

            # Fragment ul/left should not be translated horizontally but image has to be
            # padded horizontally.
            trans_x = 0
            expand_x = np.shape(other_fragment.tform_image)[1] + eps

            # Perform vertical translation depending on size of neighbouring fragment.
            if np.shape(self.tform_image)[0] < np.shape(other_fragment.tform_image)[0]:
                trans_y = np.shape(other_fragment.tform_image)[0] - np.shape(self.tform_image)[0]
            else:
                trans_y = 0

            # Get output shape. This should be larger than original image due to
            # translation
            output_shape = (
                np.shape(self.tform_image)[0] + trans_y,
                np.shape(self.tform_image)[1] + expand_x,
            )

        elif self.final_orientation in ["ur", "right"]:

            # Fragments ur/right should be translated horizontally
            trans_x = np.shape(other_fragment.tform_image)[1] + eps

            # Perform vertical translation depending on size of neighbouring fragment.
            if np.shape(self.tform_image)[0] < np.shape(other_fragment.tform_image)[0]:
                trans_y = np.shape(other_fragment.tform_image)[0] - np.shape(self.tform_image)[0]
            else:
                trans_y = 0

            # Get output shape. This should be larger than original image due to
            # translation
            output_shape = (
                np.shape(self.tform_image)[0] + trans_y,
                np.shape(self.tform_image)[1] + trans_x,
            )

        elif self.final_orientation == "ll":

            # Fragment LL should not be translated horizontally/vertically but image
            # has to be padded horizontally.
            trans_x = 0
            expand_x = np.shape(other_fragment.tform_image)[1] + eps
            trans_y = 0

            # Perform vertical translation depending on size of neighbouring fragment.
            if np.shape(self.tform_image)[0] < np.shape(other_fragment.tform_image)[0]:
                expand_y = np.shape(other_fragment.tform_image)[0] - np.shape(self.tform_image)[0]
            else:
                expand_y = 0

            # Get output shape. This should be larger than original image due to
            # translation
            output_shape = (
                np.shape(self.tform_image)[0] + expand_y,
                np.shape(self.tform_image)[1] + expand_x,
            )

        elif self.final_orientation == "lr":

            # Fragment LR should be translated horizontally
            trans_x = np.shape(other_fragment.tform_image)[1] + eps
            trans_y = 0

            # Perform vertical translation depending on size of neighbouring fragment.
            if np.shape(self.tform_image)[0] < np.shape(other_fragment.tform_image)[0]:
                expand_y = np.shape(other_fragment.tform_image)[0] - np.shape(self.tform_image)[0]
            else:
                expand_y = 0

            # Get output shape. This should be larger than original image due to
            # translation
            output_shape = (
                np.shape(self.tform_image)[0] + expand_y,
                np.shape(self.tform_image)[1] + trans_x,
            )

        elif self.final_orientation == "top":

            # Top fragment should not be translated vertically but has to be
            # padded vertically
            trans_y = 0
            expand_y = np.shape(other_fragment.tform_image_local)[0] + eps

            # Perform horizontal translation depending on size of the fused LL/LR
            # piece.
            if np.shape(self.tform_image_local)[1] < np.shape(other_fragment.tform_image_local)[1]:
                trans_x = (
                    np.shape(other_fragment.tform_image_local)[1]
                    - np.shape(self.tform_image_local)[1]
                )
            else:
                trans_x = 0

            # Get output shape. This should be larger than original image due to
            # translation
            output_shape = (
                np.shape(self.tform_image_local)[0] + expand_y,
                np.shape(self.tform_image_local)[1] + trans_x,
            )

        elif self.final_orientation == "bottom":

            # Bottom fragment should be translated and padded vertically
            trans_y = np.shape(other_fragment.tform_image_local)[0] + eps

            # Perform horizontal translation depending on size of the fused LL/LR
            # piece.
            if np.shape(self.tform_image_local)[1] < np.shape(other_fragment.tform_image_local)[1]:
                trans_x = (
                    np.shape(other_fragment.tform_image_local)[1]
                    - np.shape(self.tform_image_local)[1]
                )
            else:
                trans_x = 0

            # Get output shape. This should be larger than original image due to
            # translation
            output_shape = (
                np.shape(self.tform_image_local)[0] + trans_y,
                np.shape(self.tform_image_local)[1] + trans_x,
            )

        if self.final_orientation in ["left", "right", "top", "bottom"]:
            self.output_shape = output_shape
            self.tform_image = copy.copy(self.tform_image)

        # Apply transformation. Output shape is defined such that horizontally aligned
        # fragments can be added elementwise.
        self.tform_image_local = warp_image(
            src=self.tform_image,
            center=self.image_center_pre,
            rotation=0,
            translation=(trans_x, trans_y),
            output_shape=output_shape,
        )
        self.image_center_local = warp_2d_points(
            src=self.image_center_pre,
            center=self.image_center_pre,
            rotation=0,
            translation=(trans_x, trans_y),
        )

        # Save transformation
        self.trans_x = trans_x
        self.trans_y = trans_y

        return

    def get_tformed_images_total(self, fragments):
        """
        Method to compute the transform necessary to align all pieces. This is
        basically a vertical concatenation where all fragments are padded such that
        they will have the same shape.
        """

        # Get the complementary fragment
        other_fragment_name = self.complementary_fragments_total[self.final_orientation]
        other_fragment = next(
            (i for i in fragments if i.final_orientation == other_fragment_name), None
        )

        # Epsilon value may be necessary to ensure no overlap between images.
        eps = 1

        if self.final_orientation in ["ul", "ur"]:

            # Fragments UL/UR should not be translated vertically but have to be
            # padded vertically
            trans_y = 0
            expand_y = np.shape(other_fragment.tform_image_local)[0] + eps

            # Perform horizontal translation depending on size of the fused LL/LR
            # piece.
            if np.shape(self.tform_image_local)[1] < np.shape(other_fragment.tform_image_local)[1]:
                trans_x = (
                    np.shape(other_fragment.tform_image_local)[1]
                    - np.shape(self.tform_image_local)[1]
                )
            else:
                trans_x = 0

            # Get output shape. This should be larger than original image due to
            # translation
            output_shape = (
                np.shape(self.tform_image_local)[0] + expand_y,
                np.shape(self.tform_image_local)[1] + trans_x,
            )

        elif self.final_orientation in ["ll", "lr"]:

            # Fragments LL/LR should be translated and padded vertically
            trans_y = np.shape(other_fragment.tform_image_local)[0] + eps

            # Perform horizontal translation depending on size of the fused LL/LR
            # piece.
            if np.shape(self.tform_image_local)[1] < np.shape(other_fragment.tform_image_local)[1]:
                trans_x = (
                    np.shape(other_fragment.tform_image_local)[1]
                    - np.shape(self.tform_image_local)[1]
                )
            else:
                trans_x = 0

            # Get output shape. This should be larger than original image due to
            # translation
            output_shape = (
                np.shape(self.tform_image_local)[0] + trans_y,
                np.shape(self.tform_image_local)[1] + trans_x,
            )

        # Apply transformation. Output shape is defined such that now all pieces
        # can be added elementwise to perform the reconstruction.
        self.tform_image_global = warp_image(
            src=self.tform_image_local,
            center=self.image_center_local,
            rotation=0,
            translation=(trans_x, trans_y),
            output_shape=output_shape,
        )
        self.tform_image = copy.deepcopy(self.tform_image_global)

        # Save transformation. This ensures that the transformation can be reused for
        # higher resolutions.
        self.trans_x += trans_x
        self.trans_y += trans_y
        self.output_shape = np.shape(self.tform_image_global)

        return

    def get_tformed_images(self, tform):
        """
        Method to apply the previously acquired transformation to align all images.
        """

        # Extract initial transformation values
        trans_x, trans_y, angle, center, output_shape = tform

        # Get rotation matrix and update it with translation
        rot_mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        rot_mat[0, 2] += trans_x
        rot_mat[1, 2] += trans_y

        # Warp images
        self.colour_image = cv2.warpAffine(
            src=self.colour_image_original, M=rot_mat, dsize=output_shape[::-1]
        )
        self.tform_image = cv2.warpAffine(
            src=self.gray_image_original, M=rot_mat, dsize=output_shape[::-1]
        )
        self.mask = cv2.warpAffine(src=self.mask_original, M=rot_mat, dsize=output_shape[::-1])

        # Save image center after transformation. This will be needed for the cost
        # function later on.
        mask_contour, _ = cv2.findContours(self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        mask_contour = np.squeeze(max(mask_contour, key=cv2.contourArea))

        # Get centerpoint of the contour
        self.image_center_peri = tuple(np.round(np.mean(mask_contour, axis=0), 1))

        return

    def get_image_center(self):
        """
        Custom function to compute the center point of the fragment BEFORE
        transformation. This point is essential for defining the final transformation
        as a single transformation matrix.
        """

        # Get mask contour
        self.mask_contour, _ = cv2.findContours(self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        self.mask_contour = np.squeeze(max(self.mask_contour, key=cv2.contourArea))

        # Compute centerpoint
        self.image_center_pre = tuple([int(i) for i in np.mean(self.mask_contour, axis=0)])

        return

    def compute_edges(self):
        """
        Method to obtain edge AB and AD which are defined as the points on the mask
        that go from A->B and D->A. Recall that corner A is the corner closest to the
        prostate center and that other fragments are named in clockwise direction.
        """

        # With 2 fragments we only have 1 stitching edge.
        if self.num_fragments == 2:

            if self.mask_corner_a_idx < self.mask_corner_b_idx:
                option_a = list(self.cnt[self.mask_corner_a_idx : self.mask_corner_b_idx])
                option_b = list(self.cnt[self.mask_corner_a_idx :]) + list(
                    self.cnt[: self.mask_corner_b_idx]
                )
            else:
                option_a = list(self.cnt[self.mask_corner_b_idx : self.mask_corner_a_idx])
                option_b = list(self.cnt[self.mask_corner_b_idx :]) + list(
                    self.cnt[: self.mask_corner_a_idx]
                )

            edge_ab = option_a if len(option_a) < len(option_b) else option_b
            edge_ad = None

        # With 4 fragments we have to take the other cornerpoints into account.
        elif self.num_fragments == 4:

            # Define edge AB as part of the contour that goes from corner A to corner B
            if self.mask_corner_a_idx < self.mask_corner_b_idx:
                edge_ab = list(self.cnt[self.mask_corner_a_idx : self.mask_corner_b_idx])
            else:
                edge_ab = list(self.cnt[self.mask_corner_a_idx :]) + list(
                    self.cnt[: self.mask_corner_b_idx]
                )

            edge_ab = np.array(edge_ab)

            # Define edge AD as part of the contour that goes from corner D to corner A
            if self.mask_corner_a_idx > self.mask_corner_d_idx:
                edge_ad = list(self.cnt[self.mask_corner_d_idx : self.mask_corner_a_idx])
            else:
                edge_ad = list(self.cnt[self.mask_corner_d_idx :]) + list(
                    self.cnt[: self.mask_corner_a_idx]
                )

            edge_ad = np.array(edge_ad)

        return edge_ab, edge_ad

    def get_edges(self):
        """
        Custom method to specify the horizontal and vertical edge of a fragment.
        """

        # Get list with corners from A -> D. Note that we compute a new contour over
        # the rotated image rather than reusing the old contour.
        self.get_bbox_corners(image=self.tform_image)

        # Get edge AB and AD
        edge_ab, edge_ad = self.compute_edges()

        # Define which edge is horizontal and which edge is vertical based on orientation
        fragment_names = ["ul", "ur", "ll", "lr", "left", "right", "top", "bottom"]
        h_edges = [edge_ab, edge_ad, edge_ad, edge_ab, None, None, edge_ab, edge_ab]
        v_edges = [edge_ad, edge_ab, edge_ab, edge_ad, edge_ab, edge_ab, None, None]
        edge_dict = dict()

        for fragment, h, v in zip(fragment_names, h_edges, v_edges):
            edge_dict[fragment] = [h, v]

        h_edge, v_edge = edge_dict[self.final_orientation]
        if h_edge is not None:
            self.h_edge = h_edge
        if v_edge is not None:
            self.v_edge = v_edge

        return

    def fit_theilsen_lines(self):
        """
        Custom method to fit a Theil Sen estimator to an edge. A Theil-Sen estimator is
        used since this method is very robust to noise, resulting in a very robust
        approximation of the edge. The Theil-Sen estimator can however get computationally
        expensive on higher resolutions, which is alleviated by sampling points along
        the edge rather than taking all the edge coordinates into account.
        """

        # Initiate theilsen instance, one for vertical and one for horizontal line
        theilsen_h = TheilSenRegressor()
        theilsen_v = TheilSenRegressor()

        ### In case of horizontal edge use regular X/Y convention
        # Get X/Y coordinates of horizontal edge. For higher resolutions we don't use
        # all edge points as this can become quite computationally expensive.
        sample_rate = int(np.ceil(self.res * 20))
        eps = 1e-5

        if hasattr(self, "h_edge"):
            x_edge = np.array([i[0] for i in self.h_edge])
            x_edge = x_edge[:, np.newaxis]
            x_edge = x_edge[::sample_rate]
            x = x_edge[
                :,
            ]
            y_edge = np.array([i[1] for i in self.h_edge])
            y_edge = y_edge[::sample_rate]

            # Fit coordinates
            theilsen_h.fit(x_edge, y_edge)
            y_pred = theilsen_h.predict(x)

            # Calculate slope and intercept of line
            x_dif = x[-1].item() - x[0].item()
            y_dif = y_pred[-1] - y_pred[0]
            slope = y_dif / (x_dif + eps)
            intercept = y_pred[0] - slope * x[0].item()

            # Get final line
            x_line = [x[0].item(), x[-1].item()]
            y_line = [x[0].item() * slope + intercept, x[-1].item() * slope + intercept]
            self.h_edge_theilsen_endpoints = np.array(
                [[x, y] for x, y in zip(x_line, y_line)], dtype=object
            )
            self.h_edge_theilsen_coords = np.array(
                [[x, y] for x, y in zip(np.squeeze(x_edge), y_pred)], dtype=object
            )

        ### In case of vertical edge we need to swap X/Y
        # Get X/Y coordinates of edge
        if hasattr(self, "v_edge"):
            x_edge = np.array([i[1] for i in self.v_edge])
            x_edge = x_edge[:, np.newaxis]
            x_edge = x_edge[::sample_rate]
            x = x_edge[:]
            y_edge = np.array([i[0] for i in self.v_edge])
            y_edge = y_edge[::sample_rate]

            # Fit coordinates
            theilsen_v.fit(x_edge, y_edge)
            y_pred = theilsen_v.predict(x)

            # Calculate slope and intercept of line
            x_dif = x[-1].item() - x[0].item()
            y_dif = y_pred[-1] - y_pred[0]
            slope = y_dif / (x_dif + eps)
            intercept = y_pred[0] - slope * x[0].item()

            # Get final line
            x_line = [x[0].item() * slope + intercept, x[-1].item() * slope + intercept]
            y_line = [x[0].item(), x[-1].item()]
            self.v_edge_theilsen_endpoints = np.array(
                [[x, y] for x, y in zip(x_line, y_line)], dtype=object
            )
            self.v_edge_theilsen_coords = np.array(
                [[x, y] for x, y in zip(y_pred, np.squeeze(x_edge))], dtype=object
            )

        return
