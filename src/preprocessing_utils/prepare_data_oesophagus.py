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
    def __init__(self, image_path, save_dir, level):

        assert os.path.isfile(image_path), "mask path does not exist"
        assert isinstance(level, int), "level must be an integer"

        self.image_path = image_path
        self.mask_path = self.image_path.replace("image", "mask")
        self.filename = os.path.basename(self.image_path)
        self.save_dir = save_dir
        self.new_level = level

        self.opener = mir.MultiResolutionImageReader()

        return

    def load(self):
        """
        Function to load and downsample the raw mask.
        """

        # load raw image
        self.raw_image = self.opener.open(self.image_path)
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
            print("magic2")

        # In case of multiple labels (should be always), we assume that the largest one
        # is the mask.
        elif num_labels > 1:
            largest_cc_label = np.argmax(stats[1:, -1]) + 1
            self.mask = ((labeled_im == largest_cc_label) * 255).astype("uint8")
            print("magic")

        # Closing operation to close some holes on the mask border
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))
        self.mask = cv2.morphologyEx(
            src=self.mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=2
        )

        # Flood fill to remove holes inside mask. The floodfill mask is required by opencv
        seedpoint = (0, 0)
        pad = 5
        self.mask = np.pad(self.mask, [[pad, pad], [pad, pad]], mode="constant", constant_values=0)
        floodfill_mask = np.zeros(
            (self.mask.shape[0] + 2, self.mask.shape[1] + 2)
        ).astype("uint8")
        _, _, self.mask, _ = cv2.floodFill(self.mask, floodfill_mask, seedpoint, 255)
        self.mask = self.mask[1+pad:-1-pad, 1+pad:-1-pad]
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

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        # Save mask
        mask_filename = self.filename.replace("image", "mask")
        mask_savefile = f"{self.save_dir}/{mask_filename}"

        if not os.path.isfile(mask_savefile):
            cv2.imwrite(mask_savefile, self.mask)
        else:
            print(f"Already saved file {self.filename.replace('image', 'mask')}")

        # Save image
        image_savefile = f"{self.save_dir}/{self.filename}"

        if not os.path.isfile(image_savefile):
            cv2.imwrite(image_savefile, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        else:
            print(f"Already saved file {self.filename}")

        return


def collect_arguments():
    """
    Command line arguments
    """

    """
    parser = argparse.ArgumentParser(
        description="Postprocess tissuemasks from tissue segmentation algorithm"
    )
    parser.add_argument(
        "--imagedir", required=True, help="dir with the images and masks"
    )
    parser.add_argument(
        "--savedir", required=True, help="dir to save the postprocessed masks"
    )
    parser.add_argument(
        "--level", required=True, type=int, help="resolution level for saving",
    )
    args = parser.parse_args()

    image_dir = args.imagedir + "/*"
    save_dir = args.savedir
    level = args.level
    """

    # """
    image_dir = "raw/*"
    save_dir = "processed"
    level = 4
    # """

    return image_dir, save_dir, level


def main():

    # Collect arguments
    image_dir, save_dir, level = collect_arguments()

    # Get all masks in the mask dir
    files = sorted([i for i in glob.glob(image_dir) if "image" in i])

    if len(files) == 0:
        raise ValueError("no files found in mask directory!")

    # Process and save mask with corresponding image
    for file in tqdm.tqdm(files[3:]):
        mask = Processor(
            image_path=file, save_dir=save_dir, level=level
        )
        mask.load()
        mask.postprocess()
        mask.save()

    return


if __name__ == "__main__":
    main()
