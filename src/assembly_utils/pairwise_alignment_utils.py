import cv2
import numpy as np
import rdp
import matplotlib.pyplot as plt
import math
import itertools
import multiresolutionimageinterface as mir
import shutil
import copy

from skimage.morphology import skeletonize
from scipy.spatial import distance
from keras_preprocessing.image import ImageDataGenerator
from collections import Counter
from sklearn.linear_model import TheilSenRegressor
from scipy.spatial import distance


class Fragment:
    """
    Class for the individual fragments. This class mainly consists of methods regarding
    image processing and holds several attributes that are required for these processing
    steps.
    """

    def __init__(self, kwargs):
        self.all_fragment_names = kwargs["fragment_names"]
        self.all_temp_fragment_names = [
            f"fragment{i}.png" for i in range(1, len(self.all_fragment_names) + 1)
        ]
        self.fragment_name = kwargs["fragment_name"]
        self.num_fragments = kwargs["n_fragments"]
        self.fragment_name_idx = int(self.fragment_name.lstrip("fragment").rstrip(".png")) - 1
        self.original_name = self.all_fragment_names[self.fragment_name_idx].split(".")[0]

        self.save_dir = kwargs["save_dir"]
        self.data_dir = kwargs["data_dir"]
        self.im_path = self.save_dir.joinpath(f"preprocessed_images/{self.fragment_name}")
        self.mask_path = self.save_dir.joinpath(f"preprocessed_masks/{self.fragment_name}")
        self.res = kwargs["resolutions"][0]
        self.bg_color = kwargs["bg_color"]

        self.landmark_level = kwargs["image_level"]
        self.n_samples = 10

        self.force_config = self.data_dir.joinpath("force_config.txt").exists()
        if self.force_config:

            self.config_dict = dict()
            with open(self.data_dir.joinpath("force_config.txt"), "r") as f:
                data = f.readlines()
                for line in data:
                    self.config_dict[line.split(":")[0]] = line.split(":")[-1].rstrip("\n")

            self.location = self.config_dict[self.original_name]

        if self.num_fragments == 4:
            self.classifier = kwargs["fragment_classifier"]

        # Check for landmark points
        if not self.save_dir.joinpath("landmarks").is_dir():
            self.save_dir.joinpath("landmarks").mkdir()

        landmark_path = self.data_dir.joinpath(f"fragment{self.fragment_name_idx+1}_coordinates.npy")
        if landmark_path.exists():
            print("Using provided landmarks")
            dst = self.save_dir.joinpath(
                "landmarks", f"fragment{self.fragment_name_idx+1}_coordinates.npy"
            )

            shutil.copyfile(landmark_path, dst)
            self.require_landmark_computation = False
        else:
            self.require_landmark_computation = True

        return

    def read_images(self):
        """
        Method to read and process the fragment image.
        """

        assert self.im_path.exists(), f"image file: {self.im_path} doesn't exist"
        assert self.mask_path.exists(), f"mask file: {self.mask_path} doesn't exist"

        # Read images and masks
        self.original_image = cv2.imread(str(self.im_path))
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.mask = cv2.imread(str(self.mask_path))
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        return

    def process_images(self):
        """
        Preprocess the images a bit for the fragment classifier.
        """

        # Get largest image size of all fragments
        temp_masks = [
            cv2.imread(str(self.save_dir.joinpath(f"preprocessed_masks/{f}")))
            for f in self.all_temp_fragment_names
        ]
        max_shape = np.max([m.shape[:2] for m in temp_masks])

        # Use this to pad the image to a square - this is required for later
        xpad = (max_shape - self.mask.shape[0]) / 2
        ypad = (max_shape - self.mask.shape[1]) / 2

        self.original_image = np.pad(
            self.original_image,
            [
                [int(np.floor(xpad)), int(np.ceil(xpad))],
                [int(np.floor(ypad)), int(np.ceil(ypad))],
                [0, 0],
            ],
            constant_values=255,
        )
        self.mask = np.pad(
            self.mask,
            [[int(np.floor(xpad)), int(np.ceil(xpad))], [int(np.floor(ypad)), int(np.ceil(ypad))]],
            constant_values=0,
        )

        # Get 'offset' rotation of the image
        cnt, _ = cv2.findContours(
            (self.mask * 255).astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        cnt = np.squeeze(max(cnt, key=cv2.contourArea))
        bbox = cv2.minAreaRect(cnt)

        # Rotate the images such that the stitch edges are approximately perpendicular to
        # the XY-axis. The fragment classifier was also trained on these images.
        center = np.mean(cnt, axis=0)
        self.angle = bbox[2] if bbox[2] < 45 else bbox[2] - 90

        # Pad the image a bit for better rotation
        pad = int(self.original_image.shape[0] * 0.5)
        center += pad
        rot_mat = cv2.getRotationMatrix2D(center=center, angle=self.angle, scale=1)

        self.original_image = np.pad(
            self.original_image,
            [[pad, pad], [pad, pad], [0, 0]],
            constant_values=255,
        )
        self.mask = np.pad(
            self.mask,
            [[pad, pad], [pad, pad]],
            constant_values=0,
        )

        self.original_image = cv2.warpAffine(
            self.original_image,
            rot_mat,
            self.original_image.shape[:-1],
            borderValue=(255, 255, 255)
        )
        self.mask = cv2.warpAffine(
            self.mask,
            rot_mat,
            self.mask.shape,
            borderValue=(0, 0, 0)
        )
        pad = int(0.8*pad)
        self.original_image = self.original_image[pad:-pad, pad:-pad]
        self.mask = self.mask[pad:-pad, pad:-pad]

        ### EXPERIMENTAL ###
        # Some scanners use a slightly gray background for areas without tissue. This
        # may confuse the fragment classifier since the model was trained on images with
        # a perfect white background [255, 255, 255]. Hence, gray colours are
        # converted to white with this step.
        self.original_image_hsv = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
        self.original_image_new = copy.copy(self.original_image)
        sat_thres = 15
        self.original_image_new[self.original_image_hsv[:, :, 1] < sat_thres] = 255
        ### EXPERIMENTAL ###

        # Make smaller version for classifier
        if self.num_fragments == 4:
            self.model_image = cv2.resize(
                self.original_image_new, (self.classifier.model_size, self.classifier.model_size)
            )

        return

    def classify_stitch_edges(self):
        """
        Method to obtain stitch edges using the EfficientNet classifier.

        Label definitions:
        1 - upper right
        2 - lower right
        3 - lower left
        4 - upper left
        """

        if self.num_fragments == 4:

            if self.force_config:
                complementary_config = {
                    "ul": "lr",
                    "ur": "ll",
                    "lr": "ul",
                    "ll": "ur"
                }
                name_to_label = {
                    "ur" : 1,
                    "lr" : 2,
                    "ll" : 3,
                    "ul" : 4
                }

                complementary_label = complementary_config[self.location.lower()]
                self.stitch_edge_label = name_to_label[complementary_label]

            else:

                # Make a pseudo ensemble of the 8 variations of this image
                flip = ["hor", "hor", "ver", "ver", "both", "both", "none", "none"]
                rot = [0, 90, 0, 90, 0, 90, 0, 90]
                true_labels = []
                tform_images = []

                # Get transformed images
                for r, f in zip(rot, flip):
                    tform_im = self.classifier.transform_image(image=self.model_image, rot=r, flip=f)
                    tform_images.append(tform_im[np.newaxis, :])

                # Put batch of images in datagenerator for fast inference
                stacked_images = np.vstack(tform_images)
                datagen = ImageDataGenerator().flow(x=stacked_images, shuffle=False)
                pred = self.classifier.model.predict(datagen)
                pred = np.argmax(pred, axis=1) + 1
                pred = pred.tolist()

                # Convert back to original label without flipping/rotating
                for l, r, f in zip(pred, rot, flip):
                    tformed_label = self.classifier.transform_label(label=l, rot=r, flip=f)
                    true_labels.append(tformed_label)

                # Get majority vote
                counter = Counter(true_labels)
                self.stitch_edge_label = counter.most_common(1)[0][0]

        return

    def save_images(self):
        """
        Method to save the images to be used in Jigsawnet. Note that the exact place
        in the script to save the image matters, since the transformation matrix
        acquired later is strictly bound to the dimensions of the image.
        """

        # Save image
        self.original_image = self.mask[:, :, np.newaxis] / 255 * self.original_image
        self.original_image = self.original_image.astype("uint8")

        cv2.imwrite(
            str(self.save_dir.joinpath("configuration_detection", self.fragment_name)),
            cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR),
        )

        return

    def get_stitch_edges(self):
        """
        Method to get some edges for matching the fragments. The method can be summarized
        as follows:
        1. Get the tissue contour
        2. Create large bounding box around contour
        3. Find closest point on contour from each bounding box corner
        4. Use the predicted stitch edge side to compute the stitching edge coordinates
        """

        ### Step 1 ###
        # Obtain contour from mask
        self.cnt, _ = cv2.findContours(
            (self.mask * 255).astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        self.cnt = np.squeeze(max(self.cnt, key=cv2.contourArea))
        self.cnt = self.cnt[::-1]

        # Possibly apply RDP algorithm for simpler representation of contour
        # self.rdp_epsilon = 1
        # self.cnt_rdp = rdp.rdp(self.cnt, epsilon=self.rdp_epsilon, algo="rec").astype("int")
        self.cnt_rdp = copy.copy(self.cnt)

        debug_this = False
        if debug_this:
            plt.figure()
            plt.imshow(self.mask, cmap="gray")
            plt.scatter(self.cnt[:, 0], self.cnt[:, 1], c='r')
            plt.scatter(self.cnt_rdp[:, 0], self.cnt_rdp[:, 1], c='g')
            plt.show()

        ### Step 2 ###
        # Get bbox and its corners
        bbox = cv2.minAreaRect(self.cnt_rdp)
        bbox_corners = cv2.boxPoints(bbox)

        # Compute distance to center
        bbox_center = np.mean(bbox_corners, axis=0)
        bbox_corner_dist = bbox_center - bbox_corners

        # Expand the bbox for a bit more robust point selection on the contour. We expand
        # the bbox in only one direction or uniformly, depending on whether we are dealing
        # with 2 or 4 fragments. These choices were based on empirical observations.
        if hasattr(self, "location") and self.num_fragments == 2:
            if self.location in ["left", "right"]:
                bbox_corners_expansion = -np.vstack([bbox_corner_dist[:, 0]*5, np.zeros_like(
                    bbox_corner_dist[:, 0])]).T

            elif self.location in ["top", "bottom"]:
                bbox_corners_expansion = -np.vstack([np.zeros_like(bbox_corner_dist[:, 0]),
                                                    [bbox_corner_dist[:, 1]*5]]).T

        # Case of 4 fragments expand uniformly.
        else:
            expansion = bbox_center / 2
            expand_direction = np.array([list(i < 0) for i in bbox_corner_dist])
            bbox_corners_expansion = (expand_direction == True) * expansion + (
                expand_direction == False
            ) * -expansion

        new_bbox_corners = bbox_corners + bbox_corners_expansion

        ### Step 3 ###
        # Get closest point on contour as seen from bbox corners
        distances = distance.cdist(new_bbox_corners, self.cnt_rdp)
        indices = np.argmin(distances, axis=1)
        cnt_corners = np.array([list(self.cnt_rdp[i, :]) for i in indices])
        self.cnt_corners = sort_counterclockwise(cnt_corners)
        self.cnt_corners_loop = np.vstack([self.cnt_corners, self.cnt_corners])

        ### Step 4 ###
        # Identify upper left corner as point closest to origin
        ul_cnt_corner_idx = np.argmin(np.sum(self.cnt_corners, axis=1) ** 2)
        cnt_fragments = []

        # Step 4a - scenario with 2 fragments
        if self.num_fragments == 2:

            if hasattr(self, "location"):

                if self.location == "left":
                    start_corner = self.cnt_corners_loop[ul_cnt_corner_idx + 1]
                    end_corner = self.cnt_corners_loop[ul_cnt_corner_idx + 2]
                elif self.location == "top":
                    start_corner = self.cnt_corners_loop[ul_cnt_corner_idx + 2]
                    end_corner = self.cnt_corners_loop[ul_cnt_corner_idx + 3]
                elif self.location == "right":
                    start_corner = self.cnt_corners_loop[ul_cnt_corner_idx + 3]
                    end_corner = self.cnt_corners_loop[ul_cnt_corner_idx]
                elif self.location == "bottom":
                    start_corner = self.cnt_corners_loop[ul_cnt_corner_idx]
                    end_corner = self.cnt_corners_loop[ul_cnt_corner_idx + 1]


            else:
                # Compute distance from contour corners to surrounding bbox
                dist_cnt_corner_to_bbox = np.min(
                    distance.cdist(self.cnt_corners, new_bbox_corners), axis=1
                )
                dist_cnt_corner_to_bbox_loop = np.hstack([dist_cnt_corner_to_bbox] * 2)

                # Get the pair of contour corners with largest distance to bounding box, this should
                # be the outer point of the contour. Stitch edge is then located opposite.
                dist_per_cnt_corner_pair = [
                    dist_cnt_corner_to_bbox_loop[i] ** 2 + dist_cnt_corner_to_bbox_loop[i + 1] ** 2
                    for i in range(4)
                ]
                max_dist_corner_idx = np.argmax(dist_per_cnt_corner_pair)

                # Get location and indices of these contour corners
                start_corner = self.cnt_corners_loop[max_dist_corner_idx + 2]
                end_corner = self.cnt_corners_loop[max_dist_corner_idx + 3]

            start_idx = np.argmax((self.cnt_rdp == start_corner).all(axis=1))
            end_idx = np.argmax((self.cnt_rdp == end_corner).all(axis=1)) + 1

            # Account for indexing of the contour
            if end_idx > start_idx:
                cnt_fragments = [self.cnt_rdp[start_idx:end_idx]]
            else:
                cnt_fragments = [np.vstack([self.cnt_rdp[start_idx:], self.cnt_rdp[:end_idx]])]

        # Step 4b - scenario with 4 fragments
        elif self.num_fragments == 4:

            # Loop over both stitch edges
            for i in range(2):
                # Get starting point and end point of a stitch edge
                start_corner = self.cnt_corners_loop[
                    ul_cnt_corner_idx + self.stitch_edge_label - 1 + i
                ]
                end_corner = self.cnt_corners_loop[ul_cnt_corner_idx + self.stitch_edge_label + i]

                start_idx = np.argmax((self.cnt_rdp == start_corner).all(axis=1))
                end_idx = np.argmax((self.cnt_rdp == end_corner).all(axis=1)) + 1

                if end_idx > start_idx:
                    cnt_fragment = self.cnt_rdp[start_idx:end_idx]
                else:
                    cnt_fragment = np.vstack([self.cnt_rdp[start_idx:], self.cnt_rdp[:end_idx]])
                cnt_fragments.append(cnt_fragment)

        self.cnt_fragments = cnt_fragments

        debug = False
        if debug:
            colours = ["g", "r"]
            plt.figure()
            plt.title("Stitch edge extraction from RDP")
            plt.imshow(self.mask, cmap="gray")
            for c, cnt in zip(colours, self.cnt_fragments):
                plt.scatter(cnt[:, 0], cnt[:, 1], c=c)
            plt.show()

        return

    def save_orientation(self):
        """
        Function to save the image orientation such that it matches the other fragment.
        """

        complementary_config = {
            "top": "bottom",
            "right": "left",
            "bottom": "top",
            "left": "right"
        }
        clockwise_config = list(complementary_config.keys())

        # Determine the configuration of the fragment
        center = np.mean(self.cnt, axis=0)
        avg_x, avg_y = np.mean(self.cnt_fragments[0], axis=0)
        std_x, std_y = np.std(self.cnt_fragments[0], axis=0)

        if std_x < std_y:
            config = "right" if avg_x < center[0] else "left"
        else:
            config = "bottom" if avg_y < center[1] else "top"

        # Write location solution txt file. We use this later to load in the fragments
        location_solution = self.save_dir.joinpath(
            "configuration_detection", "location_solution.txt"
        )

        # In case of a forced configuration, take this into account and rotate the images
        # accordingly.
        if self.force_config:
            # Get complementary configuration and find required rotation
            required_config = list(self.config_dict.values())[self.fragment_name_idx]
            required_config_idx = clockwise_config.index(required_config)
            config_idx = clockwise_config.index(config)
            idx_diff = required_config_idx - config_idx
            self.rot_k = idx_diff if idx_diff > 0 else idx_diff + 4

            # Save rotated images
            self.original_image = np.rot90(self.original_image, k=-self.rot_k)
            cv2.imwrite(
                str(self.save_dir.joinpath("configuration_detection", self.fragment_name)),
                cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR),
            )

            # Save forced configuration in file
            if self.fragment_name == "fragment1.png":
                with open(location_solution, "w") as f:
                    f.write(f"{self.fragment_name}:{required_config}")
            else:
                with open(location_solution, "a") as f:
                    f.write(f"\n{self.fragment_name}:{required_config}")

        # If there is no forced configuration, keep the original shape of the first image
        # and just adjust the second image such that it complements the first image.
        else:
            # First fragment retains its original state
            if self.fragment_name == "fragment1.png":
                with open(location_solution, "w") as f:
                    f.write(f"{self.fragment_name}:{config}")

                self.rot_k = 0

            # Second fragment needs to be rotated such that it matches the config
            # of the first fragment
            else:

                # Get config of first fragment
                with open(location_solution, "r") as f:
                    data = f.readlines()
                    prev_config = data[0].split(":")[-1]

                # Get complementary configuration and find required rotation
                required_config = complementary_config[prev_config]
                required_config_idx = clockwise_config.index(required_config)
                config_idx = clockwise_config.index(config)
                idx_diff = required_config_idx - config_idx
                self.rot_k = idx_diff if idx_diff > 0 else idx_diff + 4

                # Save new configuration of fragment
                with open(location_solution, "a") as f:
                    f.write(f"\n{self.fragment_name}:{required_config}")

                # Save rotated images
                self.original_image = np.rot90(self.original_image, k=-self.rot_k)
                cv2.imwrite(
                    str(self.save_dir.joinpath("configuration_detection", self.fragment_name)),
                    cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR),
                )

        return

    def save_landmark_points(self):
        """
        Method to save some automatically detected landmark points. These can be used
        later to compute the residual registration mismatch.
        Approximate steps:
        1. Load images and masks and get lowres version
        2. Get otsu mask and tissue segmentation mask
        3. Process tissue segmentation mask
        4. Combine masks
        5. Compute simplified contour and stitch edges
        6.
        """

        ### STEP 1 ###
        # Load raw image
        self.opener = mir.MultiResolutionImageReader()
        self.raw_image_dir = self.data_dir.joinpath("raw_images")
        self.raw_image_path = self.raw_image_dir.joinpath(
            self.all_fragment_names[self.fragment_name_idx]
        )
        self.raw_image = self.opener.open(str(self.raw_image_path))

        # Load raw mask
        self.raw_mask_dir = self.data_dir.joinpath("raw_masks")
        raw_mask_names = sorted([i.name for i in self.raw_mask_dir.iterdir()])
        self.raw_mask_path = self.raw_mask_dir.joinpath(
            raw_mask_names[self.fragment_name_idx]
        )
        self.raw_mask = self.opener.open(str(self.raw_mask_path))

        # Compute size differences between image and mask
        image_ds_level_dims = self.raw_image.getLevelDimensions(self.landmark_level)

        all_mask_dims = [self.raw_mask.getLevelDimensions(i) for i in range(
            self.raw_mask.getNumberOfLevels())]
        mask_ds_level = np.argmin([(i[0] - image_ds_level_dims[0])**2 for i in all_mask_dims])
        mask_ds_level_dims = self.raw_mask.getLevelDimensions(int(mask_ds_level))

        raw_image_dims = self.raw_image.getLevelDimensions(0)
        mask2raw_scaling = raw_image_dims[0] / mask_ds_level_dims[0]

        ### STEP 2 ###
        # Load tissue mask
        tissue_mask = self.raw_mask.getUCharPatch(
            startX=0,
            startY=0,
            width=mask_ds_level_dims[0],
            height=mask_ds_level_dims[1],
            level=int(mask_ds_level)
        )

        # Process tissue mask
        temp_pad = int(0.05 * tissue_mask.shape[0])
        tissue_mask = np.pad(
            np.squeeze(tissue_mask),
            [[temp_pad, temp_pad], [temp_pad, temp_pad]],
            mode="constant",
            constant_values=0,
        )

        # Get largest component
        num_labels, labeled_mask, stats, _ = cv2.connectedComponentsWithStats(
            tissue_mask, connectivity=8
        )
        largest_cc_label = np.argmax(stats[1:, -1]) + 1
        tissue_mask = ((labeled_mask == largest_cc_label) * 255).astype("uint8")

        # Close some small holes
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, strel, iterations=3)

        # Get floodfilled background
        seedpoint = (0, 0)
        floodfill_mask = np.zeros(
            (tissue_mask.shape[0] + 2, tissue_mask.shape[1] + 2)
        )
        floodfill_mask = floodfill_mask.astype("uint8")
        _, _, tissue_mask, _ = cv2.floodFill(
            tissue_mask,
            floodfill_mask,
            seedpoint,
            255
        )
        final_mask = (
                1 - tissue_mask[temp_pad + 1: -(temp_pad + 1), temp_pad + 1: -(temp_pad + 1)]
        )

        ### STEP 5 ###
        # Get contour
        cnt, _ = cv2.findContours(
            final_mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        cnt = np.squeeze(max(cnt, key=cv2.contourArea))[::-1]

        # Get enlarged bbox
        bbox = cv2.minAreaRect(cnt)
        bbox_points = cv2.boxPoints(bbox)
        bbox_points = sort_counterclockwise(bbox_points)
        bbox_center = np.mean(bbox_points, axis=0)
        bbox_corner_dist = bbox_center - bbox_points

        # Again expand bbox
        if hasattr(self, "location") and self.num_fragments == 2:
            if self.location in ["left", "right"]:
                bbox_corners_expansion = np.vstack([bbox_corner_dist[:, 0]*5, np.zeros_like(
                    bbox_corner_dist[:, 0])]).T

            elif self.location in ["top", "bottom"]:
                bbox_corners_expansion = np.vstack([np.zeros_like(bbox_corner_dist[:, 0]),
                                                    [bbox_corner_dist[:, 1]*5]]).T

        # Case of 4 fragments expand uniformly.
        else:
            expansion = bbox_center / 2
            expand_direction = np.array([list(i < 0) for i in bbox_corner_dist])
            bbox_corners_expansion = (expand_direction == True) * expansion + (
                expand_direction == False
            ) * -expansion
        new_bbox_points = bbox_points + bbox_corners_expansion

        # Get closest point on contour as seen from enlarged bbox
        distances = distance.cdist(new_bbox_points, cnt)
        indices = np.argmin(distances, axis=1)
        cnt_corners = np.array([list(cnt[i, :]) for i in indices])
        cnt_corners_loop = np.vstack([cnt_corners, cnt_corners])

        # Get index of upperleft bbox corner. We need this as the fragment label is encoded
        # with respect to the upperleft corner.
        bbox_points_rot, _ = warp_2d_points(bbox_points, bbox_center, self.angle, [0, 0])
        ul_cnt_corner_idx = np.argmin(np.sum(bbox_points_rot**2, axis=1))

        # Get the contours
        cnt_fragments = []
        if self.num_fragments == 2:

            if hasattr(self, "location"):

                if self.location == "left":
                    start_corner = cnt_corners_loop[ul_cnt_corner_idx + 1]
                    end_corner = cnt_corners_loop[ul_cnt_corner_idx + 2]
                elif self.location == "top":
                    start_corner = cnt_corners_loop[ul_cnt_corner_idx + 2]
                    end_corner = cnt_corners_loop[ul_cnt_corner_idx + 3]
                elif self.location == "right":
                    start_corner = cnt_corners_loop[ul_cnt_corner_idx + 3]
                    end_corner = cnt_corners_loop[ul_cnt_corner_idx]
                elif self.location == "bottom":
                    start_corner = cnt_corners_loop[ul_cnt_corner_idx]
                    end_corner = cnt_corners_loop[ul_cnt_corner_idx + 1]

            else:
                # Compute distance from contour corners to surrounding bbox
                dist_cnt_corner_to_bbox = np.min(
                    distance.cdist(cnt_corners, new_bbox_points), axis=1
                )
                dist_cnt_corner_to_bbox_loop = np.hstack([dist_cnt_corner_to_bbox] * 2)

                # Get the pair of contour corners with largest distance to bounding box, this should
                # be the outer point of the contour. Stitch edge is then located opposite.
                dist_per_cnt_corner_pair = [
                    dist_cnt_corner_to_bbox_loop[i] ** 2 + dist_cnt_corner_to_bbox_loop[i + 1] ** 2
                    for i in range(4)
                ]
                max_dist_corner_idx = np.argmax(dist_per_cnt_corner_pair)

                # Get location and indices of these contour corners
                start_corner = cnt_corners_loop[max_dist_corner_idx + 2]
                end_corner = cnt_corners_loop[max_dist_corner_idx + 3]

            start_idx = np.argmax((cnt == start_corner).all(axis=1))
            end_idx = np.argmax((cnt == end_corner).all(axis=1)) + 1

            # Account for indexing of the contour
            if end_idx > start_idx:
                cnt_fragments = [cnt[start_idx:end_idx]]
            else:
                cnt_fragments = [np.vstack([cnt[start_idx:], cnt[:end_idx]])]

            # Sanity check
            plt.figure()
            plt.imshow(final_mask)
            plt.plot(cnt_fragments[0][:, 0], cnt_fragments[0][:, 1], c="r")
            plt.show()

        elif self.num_fragments == 4:
            for i in range(2):
                start_corner = cnt_corners_loop[
                    ul_cnt_corner_idx + self.stitch_edge_label - 1 + i
                    ]
                end_corner = cnt_corners_loop[ul_cnt_corner_idx + self.stitch_edge_label + i]

                start_idx = np.argmax((cnt == start_corner).all(axis=1))
                end_idx = np.argmax((cnt == end_corner).all(axis=1)) + 1

                if end_idx > start_idx:
                    cnt_fragment = cnt[start_idx:end_idx]
                else:
                    cnt_fragment = np.vstack([cnt[start_idx:], cnt[:end_idx]])
                cnt_fragments.append(interpolate_contour(cnt_fragment))

        debug = False
        if debug:
            colours = ["g", "r"]
            plt.figure()
            plt.imshow(final_mask)
            for c, cnt in zip(colours, cnt_fragments):
                plt.scatter(cnt[:, 0], cnt[:, 1], c=c)
            plt.show()

        # Sample n points along each line
        line_a_idx = np.linspace(0, len(cnt_fragments[0])-1, self.n_samples).astype("int")
        line_a = cnt_fragments[0][line_a_idx]
        line_a = (line_a * mask2raw_scaling).astype("int")

        # In case of 2 fragments, just duplicate the line. This will not impair
        # downstream tasks for median distance calculation but will prevent elaborate
        # 2/4 fragment specific landmark computation.
        if self.num_fragments == 2:
            line_b = copy.copy(line_a)
        else:
            line_b_idx = np.linspace(0, len(cnt_fragments[1])-1, self.n_samples).astype("int")
            line_b = cnt_fragments[1][line_b_idx]
            line_b = (line_b * mask2raw_scaling).astype("int")

        # Save in dict for later use
        lines = {"a" : line_a, "b" : line_b}
        np.save(
            self.save_dir.joinpath("landmarks", f"fragment"
                                                f"{self.fragment_name_idx+1}_coordinates.npy"),
            lines
        )

        if debug:
            colours = ["g", "r"]
            plt.figure()
            plt.title("Landmark extraction")
            plt.imshow(final_mask)
            for c, line in zip(colours, [line_a, line_b]):
                line = (line/mask2raw_scaling).astype("int")
                plt.scatter(line[:, 0], line[:, 1], c=c)
            plt.show()

        return


def sort_counterclockwise(points):
    """
    Function to sort a list of points in counterclockwise direction.
    """

    if not type(points) == np.ndarray:
        points = np.array(points)

    assert len(points.shape) == 2, "array must be 2-dimensional"
    assert points.shape[1] == 2, "array must be shaped as Nx2"

    # Get center of points
    center_x, center_y = np.mean(points, axis=0)
    angles = [math.atan2(y - center_y, x - center_x) for x, y in points]
    counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
    counterclockwise_points = np.array([points[i] for i in counterclockwise_indices])

    return counterclockwise_points


def interpolate_contour(contour):
    """
    Function to interpolate a contour which is represented by a set of points.

    Example:
    contour = [[0, 1], [1, 5], [2, 10]]
    new_contour = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [1, 6] etc.]
    """

    assert type(contour) == np.ndarray, "contour must be of type numpy array"
    assert len(contour.shape) == 2, "contour must be 2-dimensional"

    for i in range(len(contour) - 1):

        # Get x and y values to interpolate on
        xvals = np.array([contour[i, 0], contour[i + 1, 0]]).astype("int")
        yvals = np.array([contour[i, 1], contour[i + 1, 1]]).astype("int")

        # Create steps of size 1
        max_dif = np.max([np.abs(xvals[1] - xvals[0]), np.abs(yvals[1] - yvals[0])])
        new_xvals = np.linspace(xvals[0], xvals[1], num=max_dif).astype("int")
        new_yvals = np.linspace(yvals[0], yvals[1], num=max_dif).astype("int")

        # Get interpolated contour
        interp_contour = np.array([new_xvals, new_yvals]).T

        # Add interpolated values to new contour
        if i == 0:
            new_contour = interp_contour
        else:
            new_contour = np.vstack([new_contour, interp_contour])

    return new_contour


def warp_2d_points(src, center, rotation, translation):
    """
    Convenience function to warp a set of 2D coordinates using an affine transform.

    Input:
        - Nx2 matrix with points to warp
        - Center to rotate around
        - Angle of rotation in degrees
        - Translation in pixels

    Output:
        - Nx2 matrix with warped points
    """

    # Catch use case where only 1 coordinate pair is provided as input
    if len(np.array(src).shape) == 1:
        src = np.array(src)
        src = np.transpose(src[:, np.newaxis])

    assert (
        len(np.array(src).shape) == 2 and np.array(src).shape[-1] == 2
    ), "Input must be 2 dimensionsal and be ordered as Nx2 matrix"
    assert len(translation) == 2, "Translation must consist of X/Y component"

    # Ensure variables are in correct format
    center = tuple([int(i) for i in np.squeeze(center)])
    src = src.astype("float32")

    # Create rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center=center, angle=rotation, scale=1)
    rot_mat[0, 2] += translation[0]
    rot_mat[1, 2] += translation[1]

    # Add list of ones as pseudo third dimension to ensure proper matrix calculations
    add_ones = np.ones((src.shape[0], 1))
    src = np.hstack([src, add_ones])

    # Transform points
    tform_src = rot_mat.dot(src.T).T
    tform_src = np.round(tform_src, 1)

    return tform_src, rot_mat


def warp_image(src, center, rotation, translation, output_shape=None):
    """
    Convenience function to warp a 2D image using an affine transformation.

    Input:
        - Image to warp
        - Center to rotate around
        - Angle of rotation in degrees
        - Translation in pixels
        - Output shape of warped image

    Output:
        - Warped image
    """

    # Get output shape if it is specified. Switch XY for opencv convention
    if output_shape:
        if len(output_shape) == 2:
            output_shape = tuple(output_shape[::-1])
        elif len(output_shape) == 3:
            output_shape = tuple(output_shape[:2][::-1])

    # Else keep same output size as input image
    else:
        if len(src.shape) == 2:
            output_shape = src.shape
            output_shape = tuple(output_shape[::-1])
        elif len(src.shape) == 3:
            output_shape = src.shape[:2]
            output_shape = tuple(output_shape[::-1])

    # Ensure that shape only holds integers
    output_shape = [int(i) for i in output_shape]

    # Convert to uint8 for opencv
    if src.dtype == "float":
        src = ((src / np.max(src)) * 255).astype("uint8")

    # Ensure center is in correct format
    center = tuple([int(i) for i in np.squeeze(center)])

    # Create rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center=center, angle=rotation, scale=1)
    rot_mat[0, 2] += translation[0]
    rot_mat[1, 2] += translation[1]

    # Warp image
    tform_src = cv2.warpAffine(src=src, M=rot_mat, dsize=output_shape)

    return tform_src


def FusionImage(src, dst, transform, bg_color=[0, 0, 0]):
    """
    JigsawNet function to fuse two images.
    """

    black_bg = [0, 0, 0]
    if bg_color != black_bg:
        src[np.where((src == bg_color).all(axis=2))] = [0, 0, 0]
        dst[np.where((dst == bg_color).all(axis=2))] = [0, 0, 0]

    color_indices = np.where((dst != black_bg).any(axis=2))
    color_pt_num = len(color_indices[0])
    one = np.ones(color_pt_num)

    color_indices = list(color_indices)
    color_indices.append(one)
    color_indices = np.array(color_indices)

    transformed_lin_pts = np.matmul(transform, color_indices)
    # bounding box after transform
    try:
        dst_min_row = np.floor(np.min(transformed_lin_pts[0])).astype(int)
        dst_min_col = np.floor(np.min(transformed_lin_pts[1])).astype(int)
        dst_max_row = np.ceil(np.max(transformed_lin_pts[0])).astype(int)
        dst_max_col = np.ceil(np.max(transformed_lin_pts[1])).astype(int)
    except ValueError:
        return []  # the src or dst image has the same color with background. e.g totally black.

    # global bounding box
    src_color_indices = np.where((src != black_bg).any(axis=2))
    try:
        src_min_row = np.floor(np.min(src_color_indices[0])).astype(int)
        src_min_col = np.floor(np.min(src_color_indices[1])).astype(int)
        src_max_row = np.ceil(np.max(src_color_indices[0])).astype(int)
        src_max_col = np.ceil(np.max(src_color_indices[1])).astype(int)
    except ValueError:
        return []  # the src or dst image has the same color with background. e.g totally black.

    min_row = min(dst_min_row, src_min_row)
    max_row = max(dst_max_row, src_max_row)
    min_col = min(dst_min_col, src_min_col)
    max_col = max(dst_max_col, src_max_col)

    offset_row = -min_row
    offset_col = -min_col

    offset_transform = np.float32([[1, 0, offset_col], [0, 1, offset_row]])
    point_dst_transform = np.matmul(
        np.array([[1, 0, offset_row], [0, 1, offset_col], [0, 0, 1]]), transform
    )

    # convert row, col to opencv x,y
    img_dst_transform = np.float32(
        [
            [point_dst_transform[0, 0], point_dst_transform[1, 0], point_dst_transform[1, 2]],
            [point_dst_transform[0, 1], point_dst_transform[1, 1], point_dst_transform[0, 2]],
        ]
    )

    src_transformed = cv2.warpAffine(src, offset_transform, (max_col - min_col, max_row - min_row))
    dst_transformed = cv2.warpAffine(dst, img_dst_transform, (max_col - min_col, max_row - min_row))

    src_mask_transformed = ((src_transformed != [0, 0, 0]).all(axis=2) * 255).astype("uint8")
    dst_mask_transformed = ((dst_transformed != [0, 0, 0]).all(axis=2) * 255).astype("uint8")

    # overlap detection
    a = np.all(src_transformed == black_bg, axis=2)
    b = np.all(dst_transformed != black_bg, axis=2)
    c = np.logical_and(a, b)
    c = c.reshape((c.shape[0], c.shape[1]))
    non_overlap_indices = np.where(c)
    if len(np.where(b)[0]) == 0:
        assert (
            False and "no valid pixels in transformed dst image, please check the transform process"
        )
    else:
        overlap_ratio = 1 - len(non_overlap_indices[0]) / len(np.where(b)[0])

    # fusion
    bg_indices = np.where(a)
    src_transformed[bg_indices] = dst_transformed[bg_indices]

    offset_transform_matrix = np.float32([[1, 0, offset_row], [0, 1, offset_col], [0, 0, 1]])
    return [
        src_transformed,
        point_dst_transform,
        overlap_ratio,
        offset_transform_matrix,
        src_mask_transformed,
        dst_mask_transformed,
    ]


def check_aligned_pairs_per_fragment_combo(fragments, parameters, result_dict):
    """
    Function to plot the result of the 4 possible configurations of two fragments
    """

    # Preprocess images
    keys = list(result_dict.keys())
    raw_masks = [i[0] for i in result_dict.values()]
    raw_images = [i[1] for i in result_dict.values()]
    raw_lines = [i[2] for i in result_dict.values()]
    raw_lines_a = [i[0] for i in raw_lines]
    raw_lines_b = [i[1] for i in raw_lines]
    scores = [i[5] for i in result_dict.values()]
    images, lines_a, lines_b = [], [], []

    # Make square image
    for mask, im, line_a, line_b in zip(raw_masks, raw_images, raw_lines_a, raw_lines_b):
        r, c = np.nonzero(mask)
        im = im[np.min(r) : np.max(r), np.min(c) : np.max(c), :]

        pad = int(0.1 * im.shape[0])
        shape_diff = im.shape[0] - im.shape[1]
        pad_a = int(np.floor(np.abs(shape_diff) / 2))
        pad_b = int(np.ceil(np.abs(shape_diff) / 2))

        if shape_diff > 0:
            # Pad image to square
            im = np.pad(im, [[pad, pad], [pad + pad_a, pad + pad_b], [0, 0]])

            # Adjust line coordinates to match new images dimensions
            line_a = [[x - np.min(c) + pad + pad_a, y - np.min(r) + pad] for y, x in line_a]
            line_b = [[x - np.min(c) + pad + pad_a, y - np.min(r) + pad] for y, x in line_b]
        else:
            # Pad image to square
            im = np.pad(im, [[pad + pad_a, pad + pad_b], [pad, pad], [0, 0]])

            # Adjust line coordinates to match new images dimensions
            line_a = [[x - np.min(c) + pad, y - np.min(r) + pad + pad_a] for y, x in line_a]
            line_b = [[x - np.min(c) + pad, y - np.min(r) + pad + pad_a] for y, x in line_b]

        images.append(im)
        lines_a.append(line_a)
        lines_b.append(line_b)

    # Visualize
    plt.figure(figsize=(len(images) * 2, len(images) * 2 + 2))
    plt.suptitle(
        f"Matching pairs for '{fragments[0].fragment_name}' and \n"
        f"'{fragments[1].fragment_name}' \n",
        fontsize=16,
        fontweight="bold",
    )
    n_images = int(np.sqrt(len(images)))
    for count, vars in enumerate(zip(keys, images, lines_a, lines_b, scores)):

        key, im, line_a, line_b, score = vars
        plt.subplot(n_images, n_images, count + 1)
        plt.title(f"Combo: '{key}', score: {np.round(score, 2)}", fontsize=14)
        plt.imshow(im)

        # Plot the longer line first so it doesn't obscure the shorter line
        line_a_dist = np.sqrt(
            (line_a[0][0] - line_a[1][0]) ** 2 + (line_a[0][1] - line_a[1][1]) ** 2
        )
        line_b_dist = np.sqrt(
            (line_b[0][0] - line_b[1][0]) ** 2 + (line_b[0][1] - line_b[1][1]) ** 2
        )
        if line_a_dist > line_b_dist:
            plt.plot(np.array(line_a)[:, 0], np.array(line_a)[:, 1], c="r", linewidth=4)
            plt.plot(np.array(line_b)[:, 0], np.array(line_b)[:, 1], ":", c="b", linewidth=2)
            plt.legend(
                [
                    f"{fragments[0].fragment_name.split('.')[0]}",
                    f"{fragments[1].fragment_name.split('.')[0]}",
                ]
            )
        else:
            plt.plot(np.array(line_b)[:, 0], np.array(line_b)[:, 1], c="r", linewidth=4)
            plt.plot(np.array(line_a)[:, 0], np.array(line_a)[:, 1], ":", c="b", linewidth=2)
            plt.legend(
                [
                    f"{fragments[1].fragment_name.split('.')[0]}",
                    f"{fragments[0].fragment_name.split('.')[0]}",
                ]
            )
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        parameters["save_dir"].joinpath(
            "configuration_detection",
            "checks",
            f"matches_{fragments[0].fragment_name.rstrip('.png')}_and_{fragments[1].fragment_name.rstrip('.png')}.png",
        )
    )
    plt.close()

    return


def plot_stitch_edge_classification(fragments, parameters):
    """
    Function to plot the result of the stitch edge classification model.
    """

    label_to_name = {"1": "UR", "2": "LR", "3": "LL", "4": "UL"}

    plt.figure(figsize=(8, 8))
    plt.suptitle("Stitch edge result", fontsize=18)
    for c, f in enumerate(fragments, 1):
        plt.subplot(2, 2, c)
        plt.title(f"pred: {label_to_name[str(f.stitch_edge_label)]}")
        plt.imshow(f.original_image)
        plt.plot(f.cnt_fragments[0][:, 0], f.cnt_fragments[0][:, 1], c="r")
        plt.plot(f.cnt_fragments[1][:, 0], f.cnt_fragments[1][:, 1], c="r")
        plt.scatter(f.cnt_fragments[0][0, 0], f.cnt_fragments[0][0, 1], c="r")
        plt.scatter(f.cnt_fragments[0][-1, 0], f.cnt_fragments[0][-1, 1], c="r")
        plt.scatter(f.cnt_fragments[1][0, 0], f.cnt_fragments[1][0, 1], c="r")
        plt.scatter(f.cnt_fragments[1][-1, 0], f.cnt_fragments[1][-1, 1], c="r")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        parameters["save_dir"].joinpath(
            "configuration_detection", "fragment_classifier_results.png"
        )
    )
    plt.close()

    return


def explore_pairs(fragments, parameters):
    """
    Function to compute all possible fragment pairs. This function can be divided in the
    following steps.
    1. Compute possible matches (n*m) with n the number of stitching edges of fragment a
    and m the number of stitching edges of fragment b. For all matches:
        a. Compute the transformation matrix
        b. Verify plausible overlap
        c. Compute the matching score
        d. Compute the points of intersection
    2. Visualize the potential matches
    3. Save result in JigsawNet compatible file
    """

    ### Step 1 - compute possible matches ###
    # Get all possible fragment combinations
    fragment_names = parameters["fragment_names"]
    combinations = list(itertools.combinations(fragment_names, 2))
    all_result_dicts = []

    # Loop over combinations of fragments
    for (a_name, b_name) in combinations:

        # Get fragment class and image
        a_fragment = fragments[fragment_names.index(a_name)]
        a_fragment_image = a_fragment.original_image
        a_fragment_mask = a_fragment.mask

        b_fragment = fragments[fragment_names.index(b_name)]
        b_fragment_image = b_fragment.original_image
        b_fragment_mask = b_fragment.mask

        # Get all line fragments and indices
        a_line_fragments = a_fragment.cnt_fragments
        b_line_fragments = b_fragment.cnt_fragments
        a_line_indices = np.arange(0, len(a_line_fragments))
        b_line_indices = np.arange(0, len(b_line_fragments))

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(a_fragment_mask, cmap="gray")
        # for line in a_line_fragments:
        #     plt.scatter(line[:, 0], line[:, 1], c="r")
        # plt.subplot(122)
        # plt.imshow(b_fragment_mask, cmap="gray")
        # for line in b_line_fragments:
        #     plt.scatter(line[:, 0], line[:, 1], c="r")
        # plt.show()

        # Get possible combinations of line fragments
        line_combinations = list(itertools.product(a_line_indices, b_line_indices))
        result_dict = dict()

        ### Step 1a - compute the transformation matrix ###
        # Explore all possible combinations between individual line fragments
        for line_combo in line_combinations:

            # Get individual line fragment
            line1 = a_line_fragments[line_combo[0]]
            line2 = b_line_fragments[line_combo[1]]

            ### THEILSEN DEBUG ###

            # General theilsen init
            # sample_rate = 10
            eps = 1e-5
            theilsen_l1 = TheilSenRegressor()
            theilsen_l2 = TheilSenRegressor()

            # Get theil sen representation for line 1. First check if the line is
            # horizontal or vertical as this will determine how we handle x/y coords.
            l1_initial_dx = np.abs(line1[0, 0] - line1[-1, 0])
            l1_initial_dy = np.abs(line1[0, 1] - line1[-1, 1])
            if l1_initial_dx >= l1_initial_dy:
                x_edge = np.array([i[0] for i in line1])
                x_edge = x_edge[:, np.newaxis]
                x = x_edge[
                    :,
                ]
                y_edge = np.array([i[1] for i in line1])

                # Fit coordinates
                theilsen_l1.fit(x_edge, y_edge)
                y_pred = theilsen_l1.predict(x)

                # Calculate slope and intercept of line
                x_dif = x[-1].item() - x[0].item()
                y_dif = y_pred[-1] - y_pred[0]

                # Get final line
                line1_ts = np.array(
                    [[x, y] for x, y in zip(np.squeeze(x_edge), y_pred)], dtype=object
                )
            else:
                # Swap x/y ordering since we want to predict x for a given y for vertical lines.
                x_edge = np.array([i[1] for i in line1])
                x_edge = x_edge[:, np.newaxis]
                x = x_edge[:]
                y_edge = np.array([i[0] for i in line1])

                # Fit coordinates
                theilsen_l1.fit(x_edge, y_edge)
                y_pred = theilsen_l1.predict(x)

                # Calculate slope and intercept of line
                y_dif = x[-1].item() - x[0].item()
                x_dif = y_pred[-1] - y_pred[0]

                # Get final line
                line1_ts = np.array(
                    [[x, y] for x, y in zip(y_pred, np.squeeze(x_edge))], dtype=object
                )

            # Get center of mass and angle for computing tform
            # line1_com = np.mean(line1_ts, axis=0)
            line1_com = np.array(
                [(line1_ts[0, 0] + line1_ts[-1, 0]) / 2, (line1_ts[0, 1] + line1_ts[-1, 1]) / 2]
            )
            line1_angle = np.round(((np.arctan(y_dif / (x_dif + eps)) / np.pi) * 180), 1)

            # Get theil sen representation for line 2. Again, first check if the line is
            # horizontal or vertical as this will determine how we handle x/y coords.
            l2_initial_dx = np.abs(line2[0, 0] - line2[-1, 0])
            l2_initial_dy = np.abs(line2[0, 1] - line2[-1, 1])
            if l2_initial_dx >= l2_initial_dy:
                x_edge = np.array([i[0] for i in line2])
                x_edge = x_edge[:, np.newaxis]
                x = x_edge[
                    :,
                ]
                y_edge = np.array([i[1] for i in line2])

                # Fit coordinates
                theilsen_l2.fit(x_edge, y_edge)
                y_pred = theilsen_l2.predict(x)

                # Calculate slope and intercept of line
                x_dif = x[-1].item() - x[0].item()
                y_dif = y_pred[-1] - y_pred[0]

                # Get final line
                line2_ts = np.array(
                    [[x, y] for x, y in zip(np.squeeze(x_edge), y_pred)], dtype=object
                )
            else:
                x_edge = np.array([i[1] for i in line2])
                x_edge = x_edge[:, np.newaxis]
                x = x_edge[:]
                y_edge = np.array([i[0] for i in line2])

                # Fit coordinates
                theilsen_l2.fit(x_edge, y_edge)
                y_pred = theilsen_l2.predict(x)

                # Calculate slope and intercept of line
                y_dif = x[-1].item() - x[0].item()
                x_dif = y_pred[-1] - y_pred[0]

                # Get final line
                line2_ts = np.array(
                    [[x, y] for x, y in zip(y_pred, np.squeeze(x_edge))], dtype=object
                )

            # Get center of mass and angle for computing tform
            # line2_com = np.mean(line2_ts, axis=0)
            line2_com = np.array(
                [(line2_ts[0, 0] + line2_ts[-1, 0]) / 2, (line2_ts[0, 1] + line2_ts[-1, 1]) / 2]
            )
            line2_angle = np.round(((np.arctan(y_dif / (x_dif + eps)) / np.pi) * 180), 1)

            # Warp line 2. Make sure to account for the translation induced by rotating
            # around the origin.
            rotation = line1_angle - line2_angle
            raw_translation = line1_com - line2_com
            line2_com_after_rot, _ = warp_2d_points(
                src=line2_com, center=(0, 0), rotation=-rotation, translation=(0, 0)
            )
            line2_rot_comp = np.squeeze(line2_com_after_rot - line2_com)
            translation = raw_translation - line2_rot_comp

            line2_warp, rot_mat = warp_2d_points(
                src=line2_ts, center=(0, 0), rotation=-rotation, translation=translation
            )

            # Warp image and mask
            b_fragment_mask_warp = warp_image(
                src=b_fragment_mask, center=(0, 0), rotation=-rotation, translation=translation,
            )

            ### Step 1b - verify the transformation matrix ###
            # If there is a lot (+20% [arbitrary]) overlap between fragments, this may
            # indicate a wrong orientation of the matched contour. In this case we rotate
            # the image another 180 degrees since in this case the lines will also
            # perfectly match.
            a_mask_area = np.sum(a_fragment_mask / 255 == 1)
            ab_mask_area = np.sum((a_fragment_mask / 255 + b_fragment_mask_warp / 255) == 2)
            if ab_mask_area > int(0.2 * a_mask_area):
                rotation = line1_angle - line2_angle + 180
                line2_com_after_rot, _ = warp_2d_points(
                    src=line2_com, center=(0, 0), rotation=-rotation, translation=(0, 0)
                )
                line2_rot_comp = np.squeeze(line2_com_after_rot - line2_com)
                translation = raw_translation - line2_rot_comp

                line2_warp, rot_mat = warp_2d_points(
                    src=line2_ts, center=(0, 0), rotation=-rotation, translation=translation
                )

            b_fragment_cnt, _ = warp_2d_points(
                src=b_fragment.cnt, center=(0, 0), rotation=-rotation, translation=translation
            )

            # Get transform. Swap some values because col/row inconsistencies
            dst_transform = np.float32(
                [
                    [rot_mat[0, 0], rot_mat[1, 0], rot_mat[1, 2]],
                    [rot_mat[0, 1], rot_mat[1, 1], rot_mat[0, 2]],
                    [0, 0, 1],
                ]
            )

            # Fuse the images
            (
                fused_image,
                point_tform,
                overlap_ratio,
                offset_transform,
                src_mask,
                dst_mask,
            ) = FusionImage(a_fragment_image, b_fragment_image, dst_transform)
            fused_mask = ((fused_image != [0, 0, 0]).all(axis=2) * 255).astype("uint8")

            # Warp both lines to coordinates of fused image
            line1_warp = np.matmul(
                offset_transform, np.vstack([line1_ts[:, ::-1].T, np.ones(len(line1_ts))])
            )
            line1_warp = (line1_warp[:2, :].T).astype("int")
            line1_interp = interpolate_contour(line1_warp)

            line2_warp = np.matmul(
                point_tform, np.vstack([line2_ts[:, ::-1].T, np.ones(len(line2_ts))])
            )
            line2_warp = (line2_warp[:2, :].T).astype("int")
            line2_interp = interpolate_contour(line2_warp)

            debug_this = False
            if debug_this:
                plt.imshow(fused_mask, cmap="gray")
                plt.scatter(line1_interp[:, 1], line1_interp[:, 0], c="r")
                plt.scatter(line2_interp[:, 1], line2_interp[:, 0], c="g")
                plt.show()

            ### Step 1c - compute the matching score ###
            # Penalty component - Difference in contour lengths
            len_line1 = cv2.arcLength(curve=line1_warp, closed=False)
            len_line2 = cv2.arcLength(curve=line2_warp, closed=False)
            fitness_len_ratio = 100 * (
                (np.min([len_line1, len_line2]) / np.max([len_line1, len_line2])) ** 3
            )

            # Penalty component - Overlap between masks
            fitness_rel_overlap = 100 * (1 - overlap_ratio) ** 6

            # Penalty component - Hausdorff distance between contours
            hausdorff1, _, _ = distance.directed_hausdorff(line1_interp, line2_interp)
            hausdorff2, _, _ = distance.directed_hausdorff(line2_interp, line1_interp)
            max_hausdorff = np.max([hausdorff1, hausdorff2])
            fitness_hausdorff = 1000 / max_hausdorff

            # Compute combined fitness. Clip to [0, 400] range to prevent outliers
            # from having too large of an influence.
            fitness = int(fitness_len_ratio + fitness_rel_overlap + fitness_hausdorff)
            fitness = np.clip(fitness, 0, 400)

            ### Step 1d - Line intersection ###
            # Dilate both masks and get overlap
            strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            src_mask_dil = cv2.dilate(src_mask, strel, iterations=10)
            dst_mask_dil = cv2.dilate(dst_mask, strel, iterations=10)
            overlap_mask = (((src_mask_dil == 255) & (dst_mask_dil == 255)) * 255).astype("uint8")

            # Create skeletonization as pseudo stitch line
            # overlap_skel = cv2.ximgproc.thinning(overlap_mask)
            overlap_skel = skeletonize(overlap_mask / 255) * 1
            r, c = np.nonzero(overlap_skel)

            # Convert skeletonization coordinates back to original image coordinates
            stitch_line = np.vstack([r, c, np.ones(len(c))])
            stitch_line = np.matmul(np.linalg.inv(point_tform), stitch_line)
            stitch_line = stitch_line[:2, :].T
            stitch_line = stitch_line[:, ::-1]

            # Save results
            result_dict[f"{line_combo[0]}_{line_combo[1]}"] = [
                fused_mask,
                fused_image,
                [line1_warp, line2_warp],
                stitch_line,
                dst_transform,
                fitness,
            ]

        ### Step 2 - visualize the resulting fragment pairs ###
        # Visualize results
        check_aligned_pairs_per_fragment_combo(
            fragments=[a_fragment, b_fragment], parameters=parameters, result_dict=result_dict,
        )

        all_result_dicts.append(result_dict)

    ### Step 3 - save the results in Jigsawnet compatible format ###
    # Save all pairs in alignments.txt file for JigsawNet
    alignment_file = parameters["save_dir"].joinpath("configuration_detection", "alignments.txt")
    with open(alignment_file, "w") as file:
        for i in range(len(fragments)):
            file.write(f"Node {i}\n")

        for fragment_pair, result_dict in zip(combinations, all_result_dicts):

            node_a = fragment_names.index(fragment_pair[0])
            node_b = fragment_names.index(fragment_pair[1])

            for item in result_dict.values():
                _, _, _, stitch_line, rot_mat, fitness = item
                stitch_line = np.array(stitch_line).ravel()
                stitch_line = [np.round(float(i), 6) for i in stitch_line]
                stitch_line = "".join(c for c in str(stitch_line) if c not in "[],")
                rot_mat = [np.round(float(i), 6) for i in rot_mat.ravel()]
                rot_mat = "".join([c for c in str(rot_mat) if c not in "[],"])
                rot_mat = rot_mat.replace("\n", "")
                file.write(f"{node_a} {node_b} {fitness} {rot_mat} line {stitch_line}\n")

    # Save file with stitch edge classification
    label_to_name = {"1": "UR", "2": "LR", "3": "LL", "4": "UL"}

    stitch_edge_file = parameters["save_dir"].joinpath(
        "configuration_detection", "stitch_edges.txt"
    )
    with open(stitch_edge_file, "w") as file:
        for f in fragments:
            file.write(f"{f.fragment_name}:{label_to_name[str(f.stitch_edge_label)]}\n")

    return
