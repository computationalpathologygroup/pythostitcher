import multiresolutionimageinterface as mir
import pathlib
import argparse
import tqdm
import numpy as np
import pyvips
import matplotlib.pyplot as plt
import cv2
import copy
import json

from shapely.geometry import LineString, Point

from line_utils import interpolate_contour, apply_im_tform_to_coords


def collect_arguments():
    """
    Parse arguments formally
    """

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="convert svs to tif"
    )
    parser.add_argument(
        "--datadir", required=True, type=pathlib.Path, help="Path with the tiffs to shred"
    )
    parser.add_argument(
        "--maskdir", required=True, type=pathlib.Path, help="Path with the tissuemasks"
    )
    parser.add_argument(
        "--savedir", required=True, type=pathlib.Path, help="Path to save the shreds"
    )
    parser.add_argument(
        "--rotation", required=False, type=int, default=5, help="Random rotation of the whole"
                                                               "mount before shredding"
    )
    parser.add_argument(
        "--fragments", required=False, type=int, default=4, help="Number of fragments to shred to"
    )
    args = parser.parse_args()

    # Extract arguments
    data_dir = pathlib.Path(args.datadir)
    mask_dir = pathlib.Path(args.maskdir)
    save_dir = pathlib.Path(args.savedir)
    rotation = args.rotation
    n_fragments = args.fragments

    assert any([data_dir.is_dir(), data_dir.exists()]), "provided data location doesn't exist"
    assert mask_dir.is_dir(), "provided mask location doesn't exist"
    assert rotation in np.arange(0, 26), "rotation must be in range [0, 25]"
    assert n_fragments in [2, 4], "number of fragments must be either 2 or 4"

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    print(
        f"\nRunning job with following parameters:"
        f"\n - Data dir: {data_dir}"
        f"\n - Tissue mask dir: {mask_dir}"
        f"\n - Save dir: {save_dir}"
        f"\n - Rotation: {rotation}"
        f"\n - Number of fragments: {n_fragments}"
        f"\n"
    )

    return data_dir, mask_dir, save_dir, rotation, n_fragments


class Shredder:

    def __init__(self, case, mask_dir, save_dir, rotation, n_fragments):

        self.case = case
        self.mask_path = mask_dir.joinpath(f"{self.case.stem}.tif")
        self.savedir = save_dir.joinpath(self.case.stem)
        self.rotation = rotation
        self.n_fragments = n_fragments
        self.lowres_level = 6
        self.pad_factor = 0.3
        self.n_samples = 10
        self.noise = 20
        self.step = 50

        self.parameters = {"rotation" : self.rotation, "n_fragments" : self.n_fragments}

        if not self.savedir.is_dir():
            self.savedir.mkdir(parents=True)
            self.savedir.joinpath("raw_images").mkdir()
            self.savedir.joinpath("raw_masks").mkdir()

        return

    def load_images(self):
        """
        Load the pyramidal image and get a downsampled image from it
        """

        # Get low resolution image
        self.opener = mir.MultiResolutionImageReader()
        self.mir_image = self.opener.open(str(self.case))

        self.ds_factor = int(self.mir_image.getLevelDownsample(self.lowres_level))
        self.lowres_image_dims = self.mir_image.getLevelDimensions(self.lowres_level)
        self.lowres_image = self.mir_image.getUCharPatch(
            0,
            0,
            *self.lowres_image_dims,
            self.lowres_level
        )

        # Remove paraffin for better tissue masking later
        self.lowres_image_hsv = cv2.cvtColor(self.lowres_image, cv2.COLOR_RGB2HSV)
        sat_thres = 15
        self.sat_mask = self.lowres_image_hsv[:, :, 1] < sat_thres
        self.lowres_image[self.sat_mask] = 255

        return

    def get_mask(self):
        """
        Get the postprocessed mask of the downsampled image
        """

        # Retrieve mask
        self.lowres_mask = np.all(self.lowres_image != [255, 255, 255], axis=2)
        self.lowres_mask = (self.lowres_mask * 255).astype("uint8")

        ### Flood fill the mask to remove holes
        # 1. Get enlarged version as we want to floodfill the background
        self.temp_pad = int(0.05 * self.lowres_mask.shape[0])
        self.lowres_mask = np.pad(
            self.lowres_mask,
            [[self.temp_pad, self.temp_pad], [self.temp_pad, self.temp_pad]],
            mode="constant",
            constant_values=0,
        )

        # Slightly increase mask size, required later on
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        self.lowres_mask = cv2.dilate(self.lowres_mask, strel)

        # 2. Get floodfilled background
        seedpoint = (0, 0)
        self.floodfill_mask = np.zeros(
            (self.lowres_mask.shape[0] + 2, self.lowres_mask.shape[1] + 2)
        )
        self.floodfill_mask = self.floodfill_mask.astype("uint8")
        _, _, self.lowres_mask, _ = cv2.floodFill(
            self.lowres_mask,
            self.floodfill_mask,
            seedpoint,
            255
        )

        # 3. Convert back to foreground using array slicing and inversion
        self.lowres_mask = (
                1 - self.lowres_mask[self.temp_pad + 1: -(self.temp_pad + 1), self.temp_pad + 1: -(
                self.temp_pad + 1)]
        )

        # Get largest connected component to remove small islands
        num_labels, labeled_mask, stats, _ = cv2.connectedComponentsWithStats(
            self.lowres_mask, connectivity=8
        )
        largest_cc_label = np.argmax(stats[1:, -1]) + 1
        self.lowres_mask = ((labeled_mask == largest_cc_label) * 255).astype("uint8")

        return

    def process(self):
        """
        Method to get some image characteristics
        """

        ### First figure out the rotation of the image
        # 1. Get temporary enlarged mask
        temp_pad = int(self.pad_factor * np.min(self.lowres_mask.shape))
        temp_mask = np.pad(
            self.lowres_mask,
            [[temp_pad, temp_pad], [temp_pad, temp_pad]],
            mode="constant",
            constant_values=0,
        )

        # 2. Get largest contour
        cnt, _ = cv2.findContours(
            temp_mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        cnt = np.squeeze(max(cnt, key=cv2.contourArea))

        # 3. Get rotation angle and apply rotation
        bbox = cv2.minAreaRect(cnt)
        angle = bbox[2]
        if angle > 45:
            angle = angle-90

        # Create some extra noise in angle to create some imperfect cuts
        angle_noise = np.random.randint(-self.rotation, self.rotation)
        self.angle = int(angle + angle_noise)
        rot_center = (0, 0)
        rot_mat = cv2.getRotationMatrix2D(center=rot_center, angle=self.angle, scale=1)
        temp_mask = cv2.warpAffine(
            src=temp_mask, M=rot_mat, dsize=temp_mask.shape[::-1]
        )
        temp_mask = ((temp_mask > 128)*255).astype("uint8")

        # 4. Crop back to original size
        self.r, self.c = np.nonzero(temp_mask)
        self.lowres_mask = temp_mask[
            np.min(self.r):np.max(self.r), np.min(self.c):np.max(self.c)
        ]

        return

    def get_shred_parameters(self):
        """
        Method to get 4 equal masks
        """

        ### GET SHREDDING LINES ###

        # Get self.offset
        self.offset = 5

        # Get outer points of vertical shred line
        v_start = [int(0.5*self.lowres_mask.shape[1]), -self.offset]
        v_end = [int(0.5*self.lowres_mask.shape[1]), self.lowres_mask.shape[0]+self.offset-1]

        # Get the shredding line with some noise
        self.v_line_y = np.arange(v_start[1], v_end[1]+self.step, step=self.step)
        self.v_line_x = [v_start[0]]
        while len(self.v_line_x) < len(self.v_line_y):
            self.v_line_x.append(self.v_line_x[-1] + np.random.randint(-self.noise, self.noise))

        self.parameters["step_size"] = self.step
        self.parameters["edge_curvature"] = self.noise

        # Get outer points of horizontal shred line
        h_start = [-self.offset, int(0.5*self.lowres_mask.shape[0])]
        h_end = [self.lowres_mask.shape[1]+self.offset-1, int(0.5*self.lowres_mask.shape[0])]

        # Get the shredding line with some noise
        self.h_line_x = np.arange(h_start[0], h_end[0]+self.step, step=self.step)
        self.h_line_y = [h_start[1]]
        while len(self.h_line_y) < len(self.h_line_x):
            self.h_line_y.append(self.h_line_y[-1] + np.random.randint(-self.noise, self.noise))

        ### \ GET SHREDDING LINES ###

        ### GET INTERSECTION ###

        # Convert to shapely format
        v_line_points = [Point(x, y) for x, y in zip(self.v_line_x, self.v_line_y)]
        v_line = LineString(v_line_points)

        h_line_points = [Point(x, y) for x, y in zip(self.h_line_x, self.h_line_y)]
        h_line = LineString(h_line_points)

        # Compute intersection
        test = v_line.intersection(h_line)
        self.intersection = [int(test.x), int(test.y)]

        ### \ GET INTERSECTION ###

        ### Get final version for applying to the image
        # Interpolate for fair sampling later
        self.h_line = np.array([self.h_line_x, self.h_line_y]).T
        self.h_line_temp = interpolate_contour(self.h_line)

        # Only retain points in the mask
        self.h_line_temp = [i for i in self.h_line_temp if all(
            [0<i[1]<self.lowres_mask.shape[0], 0<i[0]<self.lowres_mask.shape[1]]
        )]
        self.h_line_interp = [i for i in self.h_line_temp if self.lowres_mask[i[1], i[0]]==255]
        self.h_line_interp = np.array(self.h_line_interp)

        # Interpolate for fair sampling later
        self.v_line = np.array([self.v_line_x, self.v_line_y]).T
        self.v_line_temp = interpolate_contour(self.v_line)

        # Only retain points in the mask
        self.v_line_temp = [i for i in self.v_line_temp if all(
            [0<i[1]<self.lowres_mask.shape[0], 0<i[0]<self.lowres_mask.shape[1]]
        )]
        self.v_line_interp = [i for i in self.v_line_temp if self.lowres_mask[i[1], i[0]]==255]
        self.v_line_interp = np.array(self.v_line_interp)

        return

    def apply_shred(self):
        """
        Apply the shredding parameters acquired in the previous step
        """

        ### APPLY SHRED PARAMETERS TO IMAGE ###

        self.shredded_mask = copy.copy(self.lowres_mask)

        # Sample points along line as markers for residual registration mismatch
        h_line_mid_idx = np.argmin(np.sum(((self.h_line_interp - self.intersection) ** 2), axis=1))
        v_line_mid_idx = np.argmin(np.sum(((self.v_line_interp - self.intersection) ** 2), axis=1))

        # Small offset in line required to ensure that sampled points are located on fragment.
        offset = 25

        # Get left part horizontal line
        h_left_sample_idx = np.linspace(offset, h_line_mid_idx, self.n_samples)
        h_left_sample_idx = h_left_sample_idx.astype("int")
        self.h_line_left = self.h_line_interp[h_left_sample_idx]

        # Get right part horizontal line
        h_right_sample_idx = np.linspace(h_line_mid_idx, len(self.h_line_interp)-1-offset,
                                         self.n_samples)
        h_right_sample_idx = h_right_sample_idx.astype("int")
        self.h_line_right = self.h_line_interp[h_right_sample_idx]

        # Get upper part vertical line
        v_upper_sample_idx = np.linspace(offset, v_line_mid_idx, self.n_samples)
        v_upper_sample_idx = v_upper_sample_idx.astype("int")
        self.v_line_upper = self.v_line_interp[v_upper_sample_idx]

        # Get lower part vertical line
        v_lower_sample_idx = np.linspace(v_line_mid_idx, len(self.v_line_interp)-1-offset,
                                         self.n_samples)
        v_lower_sample_idx = v_lower_sample_idx.astype("int")
        self.v_line_lower = self.v_line_interp[v_lower_sample_idx]

        # Divide image in either 2 or 4 fragments by etching the horizontal and
        # vertical line in the image.
        if self.n_fragments == 4:
            for line in [self.h_line_interp, self.v_line_interp]:
                self.shredded_mask[line[:, 1], line[:, 0]] = 0
        elif self.n_fragments == 2:
            self.shredded_mask[self.v_line_interp[:, 1], self.v_line_interp[:, 0]] = 0

        # Erode to ensure separation line
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.shredded_mask = cv2.erode(self.shredded_mask, strel, iterations=1)
        self.shredded_mask = ((self.shredded_mask > 128) * 255).astype("uint8")

        ### \\\ APPLY SHRED PARAMETERS TO IMAGE ###

        ### GET SHREDDED IMAGE FRAGMENTS ###
        seed_offset = 100

        if self.n_fragments == 2:
            seed_points = np.array([
                [self.intersection[0] - seed_offset, self.intersection[1]],
                [self.intersection[0] + seed_offset, self.intersection[1]]
            ])
        elif self.n_fragments == 4:
            seed_points = np.array([
                [self.intersection[0] - seed_offset, self.intersection[1] - seed_offset],
                [self.intersection[0] - seed_offset, self.intersection[1] + seed_offset],
                [self.intersection[0] + seed_offset, self.intersection[1] - seed_offset],
                [self.intersection[0] + seed_offset, self.intersection[1] + seed_offset],
            ])

        # Get individual fragments based on connected component labeling
        self.mask_fragments = []

        num_labels, self.labeled_mask, stats, _ = cv2.connectedComponentsWithStats(
            self.shredded_mask, connectivity=8
        )

        # Get lists with the sampled points
        self.all_set_a = [self.h_line_left if i[0]<self.intersection[0] else self.h_line_right for
                        i in
                     seed_points]
        self.all_set_b = [self.v_line_upper if i[1]<self.intersection[1] else self.v_line_lower for
                       i in
                     seed_points]

        for seed in seed_points:

            # Get the image based on a seed point
            label_value = self.labeled_mask[seed[1], seed[0]]
            fragment = ((self.labeled_mask == label_value) * 255).astype("uint8")
            self.mask_fragments.append(fragment)

        ### \\\ GET SHREDDED IMAGE FRAGMENTS ###

        plt.figure()
        plt.title(f"step: {self.step}, curvature: {self.noise}")
        plt.imshow(self.labeled_mask)
        for a, b in zip(self.all_set_a, self.all_set_b):
            plt.scatter(a[:, 0], a[:, 1], c="r")
            plt.scatter(b[:, 0], b[:, 1], c="r")
        plt.show()

        return

    def get_shredded_images(self):
        """
        Method to actually get the shredded images and save them
        """

        ### Process the high resolution image + mask in the EXACT same order as low res
        # 1. Pad to accomodate later rotation
        self.pyvips_image = pyvips.Image.new_from_file(str(self.case))
        output_padding = int(self.pad_factor * np.min([self.pyvips_image.width,
                                                  self.pyvips_image.height]))
        output_width = self.pyvips_image.width + 2 * output_padding
        output_height = self.pyvips_image.height + 2 * output_padding
        self.pyvips_image = self.pyvips_image.gravity(
            "centre",
            output_width,
            output_height
        )

        self.pyvips_mask = pyvips.Image.new_from_file(str(self.mask_path))
        self.pyvips_mask = self.pyvips_mask.gravity(
            "centre",
            output_width,
            output_height
        )

        # 2. Rotation through affine transformation
        rot_mat = cv2.getRotationMatrix2D(center=(0, 0), angle=self.angle, scale=1)
        rot_mat = [rot_mat[0, 0], rot_mat[0, 1], rot_mat[1, 0], rot_mat[1, 1]]
        self.pyvips_image = self.pyvips_image.affine(
            rot_mat,
            oarea=[0, 0, self.pyvips_image.width, self.pyvips_image.height]
        )
        self.pyvips_mask = self.pyvips_mask.affine(
            rot_mat,
            oarea=[0, 0, self.pyvips_mask.width, self.pyvips_mask.height]
        )

        # 3. Non zero cropping
        rmin, rmax = (
            int(self.ds_factor * np.min(self.r)),
            int(self.ds_factor * np.max(self.r)),
        )
        cmin, cmax = (
            int(self.ds_factor * np.min(self.c)),
            int(self.ds_factor * np.max(self.c)),
        )
        width = cmax - cmin
        height = rmax - rmin
        self.pyvips_image = self.pyvips_image.crop(cmin, rmin, width, height)
        self.pyvips_mask = self.pyvips_mask.crop(cmin, rmin, width, height)

        for count, vars in enumerate(zip(self.mask_fragments, self.all_set_a,
                                                           self.all_set_b), 1):
            print(f"Shredding piece {count}...")
            fragment, set_a, set_b = vars

            # Convert fragment to pyvips image
            height, width = fragment.shape
            bands = 1
            dformat = "uchar"
            fragment = (fragment / np.max(fragment)).astype("uint8")
            self.fragment = pyvips.Image.new_from_memory(
                fragment.ravel(), width, height, bands, dformat
            )

            # Get white background mask for later to correct the image
            fragment_white_bg = copy.copy(fragment)
            k = int(fragment_white_bg.shape[0] / 200)
            strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            fragment_white_bg = cv2.erode(fragment_white_bg, strel)
            fragment_white_bg = ~(fragment_white_bg * 255)
            self.fragment_white_bg = pyvips.Image.new_from_memory(
                fragment_white_bg.ravel(), width, height, bands, dformat
            )

            # Also get white foreground mask to correct the mask
            fragment_white_fg = ((~fragment_white_bg) / 255).astype("uint8")
            self.fragment_white_fg = pyvips.Image.new_from_memory(
                fragment_white_fg.ravel(), width, height, bands, dformat
            )

            # Resize to full res. Also scale landmark points
            self.fragment = self.fragment.resize(self.ds_factor)
            self.fragment_white_bg = self.fragment_white_bg.resize(self.ds_factor)
            self.fragment_white_fg = self.fragment_white_fg.resize(self.ds_factor)
            set_a = set_a * self.ds_factor
            set_b = set_b * self.ds_factor

            # Apply mask to image
            self.fragment_mask = self.pyvips_mask.multiply(self.fragment)
            self.fragment_image = self.pyvips_image.multiply(self.fragment)

            # Apply cropping for efficient saving
            r, c = np.nonzero(fragment)
            rmin, rmax = (
                int(self.ds_factor * np.min(r)),
                int(self.ds_factor * np.max(r)),
            )
            cmin, cmax = (
                int(self.ds_factor * np.min(c)),
                int(self.ds_factor * np.max(c)),
            )
            width = cmax - cmin
            height = rmax - rmin

            self.fragment_image = self.fragment_image.crop(cmin, rmin, width, height)
            self.fragment_mask = self.fragment_mask.crop(cmin, rmin, width, height)
            self.fragment_white_bg = self.fragment_white_bg.crop(cmin, rmin, width, height)
            self.fragment_white_fg = self.fragment_white_fg.crop(cmin, rmin, width, height)

            # Apply to landmark points
            set_a = np.vstack([set_a[:, 0] - cmin, set_a[:, 1] - rmin]).T
            set_b = np.vstack([set_b[:, 0] - cmin, set_b[:, 1] - rmin]).T

            # Apply random 90 degree rot for increasing stitch difficulty
            rot_k = np.random.randint(0, 4)
            self.parameters["rot_k"] = rot_k

            rot_set_a = apply_im_tform_to_coords(set_a, self.fragment_image, self.ds_factor, rot_k)
            rot_set_b = apply_im_tform_to_coords(set_b, self.fragment_image, self.ds_factor, rot_k)

            self.fragment_image = self.fragment_image.rotate(rot_k * 90)
            self.fragment_mask = self.fragment_mask.rotate(rot_k * 90)
            self.fragment_white_bg = self.fragment_white_bg.rotate(rot_k * 90)
            self.fragment_white_fg = self.fragment_white_fg.rotate(rot_k * 90)

            # Save coordinates
            rot_sets = {"a": rot_set_a, "b": rot_set_b}
            np.save(self.savedir.joinpath(f"fragment{count}_coordinates"), rot_sets)

            # Set spacing to 0.25 because it's (erroneously) not present in original file
            spacing = 0.25
            xyres = 1000 / spacing
            self.fragment_image_save = self.fragment_image.copy(xres=xyres, yres=xyres)

            # Apply white background to image. Use casting to clip values.
            self.fragment_image_save = self.fragment_image_save + self.fragment_white_bg
            self.fragment_image_save = self.fragment_image_save.cast("uchar")

            # Save image and corresponding mask
            print(f" - saving image")
            self.fragment_image_save.write_to_file(
                str(self.savedir.joinpath("raw_images", f"fragment{count}.tif")),
                tile=True,
                compression="jpeg",
                bigtiff=True,
                pyramid=True,
                Q=80,
            )

            print(f" - saving mask")
            self.fragment_mask_save = self.fragment_mask.copy(xres=xyres, yres=xyres)
            self.fragment_mask_save = self.fragment_mask_save.multiply(self.fragment_white_fg)
            self.fragment_mask_save.write_to_file(
                str(self.savedir.joinpath("raw_masks", f"fragment{count}.tiff")),
                tile=True,
                compression="jpeg",
                bigtiff=True,
                pyramid=True,
                Q=20,
            )

            # Save shredding parameters as json
            with open(self.savedir.joinpath(f"fragment{count}_shred_parameters.json"), "w") as f:
                json.dump(self.parameters, f, ensure_ascii=False)

        return


def main():
    """
    Run shredder
    """

    # Get arguments
    data_dir, mask_dir, save_dir, rotation, n_fragments = collect_arguments()

    # MULTI MODE
    
    # Get cases
    cases = sorted(list(data_dir.iterdir()))
    n1 = len(cases)
    print(f"Found {n1} cases to shred")
    cases = [i for i in cases if not save_dir.joinpath(i.name.rstrip(".tif"),
                                                       "fragment4_shred_parameters.json").exists()]
    n2 = len(cases)
    print(f"Already completed {n1-n2} cases, shredding remaining {n2} cases...")

    # Loop over cases and shred
    for case in tqdm.tqdm(cases, total=len(cases)):

        shredder = Shredder(case, mask_dir, save_dir, rotation, n_fragments)
        shredder.load_images()
        shredder.get_mask()
        shredder.process()
        shredder.get_shred_parameters()
        shredder.apply_shred()
        shredder.get_shredded_images()

    return


if __name__ == "__main__":
    main()
