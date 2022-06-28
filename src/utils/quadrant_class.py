import os
import pickle
import copy
import cv2
import numpy as np

from sklearn.linear_model import TheilSenRegressor

from .get_resname import get_resname
from .transformations import warp_2d_points, warp_image


class Quadrant:
    """
    Class for the individual quadrants. This class mainly consists of methods regarding
    image processing and holds several attributes that are required for these processing
    steps.
    """

    def __init__(self, quadrant_name, kwargs):
        self.quadrant_name = quadrant_name
        self.iteration = kwargs["iteration"]
        self.file_name = kwargs["filenames"][self.quadrant_name]
        self.patient_idx = kwargs["patient_idx"]
        self.slice_idx = kwargs["slice_idx"]

        self.resolutions = kwargs["resolutions"]
        self.nbins = kwargs["nbins"]
        self.hist_sizes = kwargs["hist_sizes"]
        self.datadir = kwargs["data_dir"]
        self.maskdir = self.datadir.replace("images", "masks")
        self.impath = os.path.join(self.datadir, self.file_name)
        self.res = self.resolutions[self.iteration]
        self.res_name = get_resname(self.res)
        self.pad_fraction = kwargs["pad_fraction"]

        return

    def read_image(self):
        """
        Method to read the quadrant image. Input images should preferably be .tiff or
        .tif files, but can probably be any format as long as it is supported by opencv.
        """

        # Image extension can be either .tif or .tiff depending on how it was preprocessed
        extensions = [".tif", ".tiff"]

        # Try different extensions
        for ext in extensions:

            impath = self.impath + ext

            if os.path.isfile(impath):
                self.original_image = cv2.imread(impath)
                self.original_image = cv2.cvtColor(
                    self.original_image, cv2.COLOR_BGR2RGB
                )

                return

        raise ValueError(f"No images found for {self.impath}")

    def preprocess_gray_image(self):
        """
        Method to preprocess original image to grayscale image.
        """

        # Convert to grayscale and resize
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        self.target_size = (
            int(np.round(self.res * np.shape(self.gray_image)[1])),
            int(np.round(self.res * np.shape(self.gray_image)[0])),
        )
        self.gray_image = cv2.resize(self.gray_image, self.target_size)

        return

    def preprocess_colour_image(self):
        """
        Method to preprocess original image to colour image. This is not actually required
        for the stitching algorithm but may be used for plotting of the result.
        """

        # Resize to smaller resolution
        self.colour_image = cv2.resize(self.original_image, self.target_size)

        return

    def segment_tissue(self):
        """
        Method to obtain tissue segmentation mask at a given resolution level. This just
        loads the mask that was previously obtained by the background segmentation algorithm.
        A later version of Pythostitcher could perhaps integrate this segmentation model.
        """

        # Mask extension can be .tif or .tiff depending on how it was preprocessed
        extensions = [".tif", ".tiff"]
        base_path = self.impath.replace("images", "masks")

        # Try different extensions
        for ext in extensions:

            mask_path = base_path + ext

            if os.path.isfile(mask_path):
                # Read mask
                self.mask = cv2.imread(mask_path)
                self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

                # Resize mask to match images
                self.mask = cv2.resize(self.mask, self.target_size)
                self.mask = (self.mask > 0.5) * 1

                return

        # Raise error if mask is not found
        raise ValueError(f"No mask found for {base_path}")

    def apply_masks(self):
        """
        Method to apply the previously obtained tissue mask to the images.
        """

        # Apply mask to gray and colour image
        self.colour_image = self.mask[:, :, np.newaxis] * self.colour_image
        self.gray_image = self.mask * self.gray_image

        # Crop the image
        c, r = np.nonzero(self.gray_image)
        cmin, cmax = np.min(c), np.max(c)
        rmin, rmax = np.min(r), np.max(r)

        self.colour_image = self.colour_image[cmin:cmax, rmin:rmax]
        self.gray_image = self.gray_image[cmin:cmax, rmin:rmax]
        self.mask = self.mask[cmin:cmax, rmin:rmax]

        # For the lowest resolution we apply a fixed padding rate. For higher resolutions
        # we compute the ideal padding based on the previous resolution such that the quadrant
        # retains the same image/pad ratio as the previous image.
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
                [
                    [self.initial_pad, self.initial_pad],
                    [self.initial_pad, self.initial_pad],
                ],
            )
            self.mask = np.pad(
                self.mask,
                [
                    [self.initial_pad, self.initial_pad],
                    [self.initial_pad, self.initial_pad],
                ],
            )
        else:
            prev_res_image_path = (
                f"../results/"
                f"{self.patient_idx}/"
                f"{self.slice_idx}/"
                f"{get_resname(self.resolutions[self.iteration-1])}/"
                f"quadrant_{self.quadrant_name}_gray.png"
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
            self.gray_image = np.pad(
                self.gray_image, [[pad[0], pad[0]], [pad[1], pad[1]]]
            )
            self.mask = np.pad(self.mask, [[pad[0], pad[0]], [pad[1], pad[1]]])

        return

    def save_quadrant(self):
        """
        Method to save the current quadrant images and the class itself with all its
        parameters.
        """

        self.quadrant_savepath = (
            f"../results/{self.patient_idx}/{self.slice_idx}/"
            f"{self.res_name}/quadrant_{self.quadrant_name}"
        )

        # Save mask, grayscale and colour image. Remove these from the class after saving
        # to prevent double saving
        save_maskfile = self.quadrant_savepath + "_mask.png"
        cv2.imwrite(
            save_maskfile,
            cv2.cvtColor((self.mask * 255).astype("uint8"), cv2.COLOR_GRAY2BGR),
        )
        del self.mask

        save_grayimfile = self.quadrant_savepath + "_gray.png"
        cv2.imwrite(
            save_grayimfile,
            cv2.cvtColor(self.gray_image.astype("uint8"), cv2.COLOR_GRAY2BGR),
        )
        del self.gray_image

        save_colourimfile = self.quadrant_savepath + "_colour.png"
        cv2.imwrite(
            save_colourimfile,
            cv2.cvtColor(self.colour_image.astype("uint8"), cv2.COLOR_RGB2BGR),
        )
        del self.colour_image
        del self.original_image

        # Save the quadrant info without the images
        with open(self.quadrant_savepath, "wb") as savefile:
            pickle.dump(self, savefile)

        return

    def load_images(self):
        """
        Method to load previously preprocessed quadrant images.
        """

        # Read all relevant images. Take into account that opencv reads images in BGR
        # rather than RGB.
        basepath_load = (
            f"../results/{self.patient_idx}/{self.slice_idx}/{self.res_name}"
        )
        self.mask = cv2.imread(
            f"{basepath_load}/quadrant_{self.quadrant_name}_mask.png"
        )
        if len(self.mask.shape) == 3:
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        self.gray_image = cv2.imread(
            f"{basepath_load}/quadrant_{self.quadrant_name}_gray.png"
        )
        self.gray_image = cv2.cvtColor(self.gray_image, cv2.COLOR_BGR2GRAY)
        self.colour_image = cv2.imread(
            f"{basepath_load}/quadrant_{self.quadrant_name}_colour.png"
        )
        self.colour_image = cv2.cvtColor(self.colour_image, cv2.COLOR_BGR2RGB)

        # Make a copy of all images
        self.mask_original = copy.deepcopy(self.mask)
        self.gray_image_original = copy.deepcopy(self.gray_image)
        self.colour_image_original = copy.deepcopy(self.colour_image)

        return

    def get_bbox_corners(self, image):
        """
        Custom method to obtain coordinates of corner A and corner C. Corner A is the
        corner of the bounding box which represents the center of the prostate. Corner C
        is the corner of the bounding box which represents the corner furthest away
        from corner A. Corners are named in clockwise direction.

        Example: upperleft quadrant
        C  >  D
        ^     v
        B  <  A

        """

        # Convert to uint8 for opencv processing
        image = (image * 255).astype(np.uint8)

        # Obtain contour from mask
        self.cnt, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        self.cnt = np.squeeze(max(self.cnt, key = cv2.contourArea))
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

        # Compute smallest distance from each corner point to any point in contour
        for corner in self.bbox_corners:
            dist_x = [np.abs(corner[0] - x_point) for x_point in x_points]
            dist_y = [np.abs(corner[1] - y_point) for y_point in y_points]
            dist = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(dist_x, dist_y)]
            mask_idx = np.argmin(dist)
            mask_corners.append(self.cnt[mask_idx])
            mask_corners_idxs.append(mask_idx)
            distances.append(np.min(dist))

        # Corner c should always be the furthest away from the mask
        corner_c_idx = np.argmax(distances)
        self.bbox_corner_c = self.bbox_corners[corner_c_idx]
        self.mask_corner_c = mask_corners[corner_c_idx]
        self.mask_corner_c_idx = mask_corners_idxs[corner_c_idx]

        # Corner a is the opposite corner and is found 2 indices further
        corner_idxs = [0, 1, 2, 3] * 2
        corner_a_idx = corner_idxs[corner_c_idx + 2]
        self.bbox_corner_a = self.bbox_corners[corner_a_idx]
        self.mask_corner_a = mask_corners[corner_a_idx]
        self.mask_corner_a_idx = mask_corners_idxs[corner_a_idx]

        # Corner b corresponds to the corner 1 index before corner c
        corner_b_idx = corner_idxs[corner_c_idx - 1]
        self.bbox_corner_b = self.bbox_corners[corner_b_idx]
        self.mask_corner_b = mask_corners[corner_b_idx]
        self.mask_corner_b_idx = mask_corners_idxs[corner_b_idx]

        # Corner d corresponds to the corner 1 index before corner a
        corner_d_idx = corner_idxs[corner_a_idx - 1]
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
            self.angle = -(90 - self.angle)
        elif self.angle < -45:
            self.angle = 90 + self.angle
        self.angle = np.round(self.angle, 1)

        # Apply rotation first
        rot_mat = cv2.getRotationMatrix2D(
            center=self.image_center_pre, angle=self.angle, scale=1
        )
        self.tform_image = cv2.warpAffine(
            src=self.gray_image, M=rot_mat, dsize=self.gray_image.shape
        )
        self.rot_mask = cv2.warpAffine(
            src=self.mask_original, M=rot_mat, dsize=self.mask_original.shape
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

        # Apply padding and save padding parameters for the transformation
        if self.quadrant_name == "UL":
            self.tform_image = np.pad(
                self.tform_image, [[self.small_pad, 0], [self.small_pad, 0]]
            )
            self.pad_trans_x = self.small_pad
            self.pad_trans_y = self.small_pad
        elif self.quadrant_name == "UR":
            self.tform_image = np.pad(
                self.tform_image, [[self.small_pad, 0], [0, self.small_pad]]
            )
            self.pad_trans_x = 0
            self.pad_trans_y = self.small_pad
        elif self.quadrant_name == "LL":
            self.tform_image = np.pad(
                self.tform_image, [[0, self.small_pad], [self.small_pad, 0]]
            )
            self.pad_trans_x = self.small_pad
            self.pad_trans_y = 0
        elif self.quadrant_name == "LR":
            self.tform_image = np.pad(
                self.tform_image, [[0, self.small_pad], [0, self.small_pad]]
            )
            self.pad_trans_x = 0
            self.pad_trans_y = 0

        return

    def get_tformed_images_local(self, quadrant):
        """
        Method to compute the transform necessary to align two horizontal pieces.
        This is basically a horizontal concatenation where both quadrants are padded
        such that they will have the same shape.
        """

        # Epsilon value is necessary to ensure no overlap between images due to rounding
        # errors.
        eps = 1

        if self.quadrant_name == "UL":

            # Quadrant UL should not be translated horizontally but image has to be
            # padded horizontally.
            trans_x = 0
            expand_x = np.shape(quadrant.tform_image)[1] + eps

            # Perform vertical translation depending on size of neighbouring quadrant.
            if np.shape(self.tform_image)[0] < np.shape(quadrant.tform_image)[0]:
                trans_y = (
                    np.shape(quadrant.tform_image)[0] - np.shape(self.tform_image)[0]
                )
            else:
                trans_y = 0

            # Get output shape. This should be larger than original image due to
            # translation
            out_shape = (
                np.shape(self.tform_image)[0] + trans_y,
                np.shape(self.tform_image)[1] + expand_x,
            )

        elif self.quadrant_name == "UR":

            # Quadrant UR should be translated horizontally
            trans_x = np.shape(quadrant.tform_image)[1] + eps

            # Perform vertical translation depending on size of neighbouring quadrant.
            if np.shape(self.tform_image)[0] < np.shape(quadrant.tform_image)[0]:
                trans_y = (
                    np.shape(quadrant.tform_image)[0] - np.shape(self.tform_image)[0]
                )
            else:
                trans_y = 0

            # Get output shape. This should be larger than original image due to
            # translation
            out_shape = (
                np.shape(self.tform_image)[0] + trans_y,
                np.shape(self.tform_image)[1] + trans_x,
            )

        elif self.quadrant_name == "LL":

            # Quadrant LL should not be translated horizontally/vertically but image
            # has to be padded horizontally.
            trans_x = 0
            expand_x = np.shape(quadrant.tform_image)[1] + eps
            trans_y = 0

            # Perform vertical translation depending on size of neighbouring quadrant.
            if np.shape(self.tform_image)[0] < np.shape(quadrant.tform_image)[0]:
                expand_y = (
                    np.shape(quadrant.tform_image)[0] - np.shape(self.tform_image)[0]
                )
            else:
                expand_y = 0

            # Get output shape. This should be larger than original image due to
            # translation
            out_shape = (
                np.shape(self.tform_image)[0] + expand_y,
                np.shape(self.tform_image)[1] + expand_x,
            )

        elif self.quadrant_name == "LR":

            # Quadrant LR should be translated horizontally
            trans_x = np.shape(quadrant.tform_image)[1] + eps
            trans_y = 0

            # Perform vertical translation depending on size of neighbouring quadrant.
            if np.shape(self.tform_image)[0] < np.shape(quadrant.tform_image)[0]:
                expand_y = (
                    np.shape(quadrant.tform_image)[0] - np.shape(self.tform_image)[0]
                )
            else:
                expand_y = 0

            # Get output shape. This should be larger than original image due to
            # translation
            out_shape = (
                np.shape(self.tform_image)[0] + expand_y,
                np.shape(self.tform_image)[1] + trans_x,
            )

        # Apply transformation. Output shape is defined such that horizontally aligned
        # quadrants can be added elementwise.
        self.tform_image_local = warp_image(
            src=self.tform_image,
            center=self.image_center_pre,
            rotation=0,
            translation=(trans_x, trans_y),
            output_shape=out_shape,
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

    def get_tformed_images_global(self, quadrant):
        """
        Method to compute the transform necessary to align all pieces. This is
        basically a vertical concatenation where all quadrants are padded such that
        they will have the same shape.
        """

        # Epsilon value may be necessary to ensure no overlap between images.
        eps = 1

        if self.quadrant_name in ["UL", "UR"]:

            # Quadrants UL/UR should not be translated vertically but have to be
            # padded vertically
            trans_y = 0
            expand_y = np.shape(quadrant.tform_image_local)[0] + eps

            # Perform horizontal translation depending on size of the fused LL/LR
            # piece.
            if (
                np.shape(self.tform_image_local)[1]
                < np.shape(quadrant.tform_image_local)[1]
            ):
                trans_x = (
                    np.shape(quadrant.tform_image_local)[1]
                    - np.shape(self.tform_image_local)[1]
                )
            else:
                trans_x = 0

            # Get output shape. This should be larger than original image due to
            # translation
            out_shape = (
                np.shape(self.tform_image_local)[0] + expand_y,
                np.shape(self.tform_image_local)[1] + trans_x,
            )

        elif self.quadrant_name in ["LL", "LR"]:

            # Quadrants LL/LR should be translated and padded vertically
            trans_y = np.shape(quadrant.tform_image_local)[0] + eps

            # Perform horizontal translation depending on size of the fused LL/LR
            # piece.
            if (
                np.shape(self.tform_image_local)[1]
                < np.shape(quadrant.tform_image_local)[1]
            ):
                trans_x = (
                    np.shape(quadrant.tform_image_local)[1]
                    - np.shape(self.tform_image_local)[1]
                )
            else:
                trans_x = 0

            # Get output shape. This should be larger than original image due to
            # translation
            out_shape = (
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
            output_shape=out_shape,
        )
        self.tform_image = self.tform_image_global

        # Save transformation. This ensures that the transformation can be reused for
        # higher resolutions.
        self.trans_x += trans_x
        self.trans_y += trans_y
        self.outshape = np.shape(self.tform_image_global)

        return

    def get_tformed_images(self, tform):
        """
        Method to apply the previously acquired transformation to align all images.
        """

        # Extract initial transformation values
        trans_x, trans_y, angle, center, out_shape = tform

        # Get rotation matrix and update it with translation
        rot_mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        rot_mat[0, 2] += trans_x
        rot_mat[1, 2] += trans_y

        # Warp images
        self.colour_image = cv2.warpAffine(
            src=self.colour_image_original, M=rot_mat, dsize=out_shape[::-1]
        )
        self.tform_image = cv2.warpAffine(
            src=self.gray_image_original, M=rot_mat, dsize=out_shape[::-1]
        )
        self.mask = cv2.warpAffine(
            src=self.mask_original, M=rot_mat, dsize=out_shape[::-1]
        )

        # Save image center after transformation. This will be needed for the cost
        # function later on.
        mask_contour, _ = cv2.findContours(
            self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        mask_contour = np.squeeze(max(mask_contour, key = cv2.contourArea))

        # Get centerpoint of the contour
        self.image_center_peri = np.mean(mask_contour, axis=0)

        return

    def get_image_center(self):
        """
        Custom function to compute the center point of the quadrant BEFORE
        transformation. This point is essential for defining the final transformation
        as a single transformation matrix.
        """

        # Get mask contour
        self.mask_contour, _ = cv2.findContours(
            self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        self.mask_contour = np.squeeze(max(self.mask_contour, key = cv2.contourArea))

        # Compute centerpoint
        self.image_center_pre = tuple(
            [int(i) for i in np.mean(self.mask_contour, axis=0)]
        )

        return

    def compute_edges(self):
        """
        Method to obtain edge AB and AD which are defined as the points on the mask
        that go from A->B and D->A. Recall that corner A is the corner closest to the
        prostate center and that other quadrants are named in clockwise direction.
        """

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
        Custom method to specify the horizontal and vertical edge of a quadrant.
        """

        # Get list with corners from A -> D. Note that we compute a new contour over
        # the rotated image rather than reusing the old contour.
        self.get_bbox_corners(image=self.tform_image)

        # Get edge AB and AD
        edge_ab, edge_ad = self.compute_edges()

        # Define which edge is horizontal and which edge is vertical based on orientation
        if self.quadrant_name == "UL":
            h_edge = edge_ab
            v_edge = edge_ad
        elif self.quadrant_name == "UR":
            h_edge = edge_ad
            v_edge = edge_ab
        elif self.quadrant_name == "LL":
            h_edge = edge_ad
            v_edge = edge_ab
        elif self.quadrant_name == "LR":
            h_edge = edge_ab
            v_edge = edge_ad
        else:
            raise NameError("Fragment must be one of UL/UR/LL/LR")

        self.h_edge = h_edge
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
        x_edge = np.array([i[0] for i in self.h_edge])
        x_edge = x_edge[:, np.newaxis]
        x_edge = x_edge[::sample_rate]
        x = x_edge[:, ]
        y_edge = np.array([i[1] for i in self.h_edge])
        y_edge = y_edge[::sample_rate]

        # Fit coordinates
        theilsen_h.fit(x_edge, y_edge)
        y_pred = theilsen_h.predict(x)

        # Calculate slope and intercept of line
        eps = 1e-5
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
        x_edge = np.array([i[1] for i in self.v_edge])
        x_edge = x_edge[:, np.newaxis]
        x_edge = x_edge[::sample_rate]
        x = x_edge[:, ]
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
