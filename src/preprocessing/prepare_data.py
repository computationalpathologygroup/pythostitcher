import multiresolutionimageinterface as mir
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import copy
import glob
import tqdm
import argparse


class Processor:
    def __init__(self, mask_path, image_dir, save_dir, level):

        assert os.path.isfile(mask_path), "mask path does not exist"
        assert os.path.isdir(image_dir), "image path does not exist"
        assert isinstance(level, int), "level must be an integer"

        self.mask_path = mask_path
        self.filename = os.path.basename(self.mask_path)
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.new_level = level

        self.opener = mir.MultiResolutionImageReader()

        return

    def load(self):
        """
        Function to load and downsample the raw mask.
        """

        # load raw image
        self.image_filename = copy.deepcopy(self.filename)
        self.image_filename = self.image_filename.replace(".tif", ".mrxs")
        self.raw_image_path = f"{self.image_dir}/{self.image_filename}"
        self.raw_image = self.opener.open(self.raw_image_path)
        assert self.raw_image.valid() == True, "Loaded image was not valid"

        # Load raw mask
        self.raw_mask = self.opener.open(self.mask_path)
        assert self.raw_mask.valid() == True, "Loaded mask was not valid"

        # Get downsampled image
        self.new_dims = self.raw_image.getLevelDimensions(self.new_level)
        self.image = self.raw_image.getUCharPatch(0, 0, *self.new_dims, self.new_level)

        # Get downsampled mask
        mask_dims = [
            self.raw_mask.getLevelDimensions(i)
            for i in range(self.raw_mask.getNumberOfLevels())
        ]
        mask_level = mask_dims.index(self.new_dims)
        self.mask = self.raw_mask.getUCharPatch(0, 0, *self.new_dims, mask_level)
        self.mask = np.squeeze(self.mask)
        self.mask = ((self.mask / np.max(self.mask)) * 255).astype("uint8")

        return

    def postprocess(self):
        """
        Function to postprocess the mask. This mainly consists
        of getting the largest component and then cleaning up this mask.
        """

        # Get information on all connected components in the mask
        num_labels, labeled_im, stats, _ = cv2.connectedComponentsWithStats(
            self.mask, connectivity=4
        )

        # Background gets counted as label, therefore an empty image will have 1 label.
        if num_labels == 1:
            raise ValueError("mask is empty")

        # In case of multiple labels (should be always), we assume that the largest one
        # is the mask.
        elif num_labels > 1:
            largest_cc_label = np.argmax(stats[1:, -1]) + 1
            self.mask = ((labeled_im == largest_cc_label) * 255).astype("uint8")

        # Closing operation to close some holes on the mask border
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))
        self.mask = cv2.morphologyEx(
            src=self.mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=2
        )

        # Flood fill to remove holes inside mask. The floodfill mask is required by opencv
        seedpoint = (0, 0)
        floodfill_mask = np.zeros(
            (self.mask.shape[0] + 2, self.mask.shape[1] + 2)
        ).astype("uint8")
        _, _, self.mask, _ = cv2.floodFill(self.mask, floodfill_mask, seedpoint, 255)
        self.mask = self.mask[1:-1, 1:-1]
        self.mask = 1 - self.mask

        assert np.sum(self.mask) > 0, "floodfilled mask is empty"

        # Crop to nonzero pixels for efficient saving
        r, c = np.nonzero(self.mask)
        self.mask = self.mask[np.min(r) : np.max(r), np.min(c) : np.max(c)]
        self.image = self.image[np.min(r) : np.max(r), np.min(c) : np.max(c)]
        self.image = self.image.astype("uint8")
        self.mask = (self.mask * 255).astype("uint8")

        return

    def save(self):
        """
        Function to save the postprocessed mask
        """

        patient_idx = self.filename.split("_")[2]
        savedir = f"{self.save_dir}/{patient_idx}"

        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        # Save mask
        mask_filename = self.filename.replace(".tif", "_MASK.tif")
        mask_savefile = f"{savedir}/{mask_filename}"

        if not os.path.isfile(mask_savefile):
            cv2.imwrite(mask_savefile, self.mask)
        else:
            print(f"Already saved file {mask_filename}")

        # Save image
        image_filename = self.image_filename.replace(".mrxs", "_IMAGE.tif")
        image_savefile = f"{savedir}/{image_filename}"

        if not os.path.isfile(image_savefile):
            cv2.imwrite(image_savefile, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        else:
            print(f"Already saved file {image_filename}")

        return


def collect_arguments():
    """
    Command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Postprocess tissuemasks from tissue segmentation algorithm"
    )
    parser.add_argument(
        "--maskdir", required=True, help="dir with the raw tissuemasks"
    )
    parser.add_argument(
        "--imagedir", required=True, help="dir with the raw images"
    )
    parser.add_argument(
        "--savedir", required=True, help="dir to save the postprocessed masks"
    )
    parser.add_argument(
        "--level", required=True, type=int, help="resolution level for saving",
    )
    args = parser.parse_args()

    mask_dir = args.maskdir + "/*"
    image_dir = args.imagedir
    save_dir = args.savedir
    level = args.level

    return mask_dir, image_dir, save_dir, level


def main():

    # Collect arguments
    mask_dir, image_dir, save_dir, level = collect_arguments()

    # Get all masks in the mask dir
    files = [i for i in glob.glob(mask_dir) if i.endswith("tif")]

    if len(files) == 0:
        raise ValueError("no files found in mask directory!")

    # Process and save mask with corresponding image
    for file in tqdm.tqdm(files):
        mask = Processor(
            mask_path=file, image_dir=image_dir, save_dir=save_dir, level=level
        )
        mask.load()
        mask.postprocess()
        mask.save()

    return


if __name__ == "__main__":
    main()
