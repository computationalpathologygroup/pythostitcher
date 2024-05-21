import multiresolutionimageinterface as mir
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchstain
from torchvision import transforms
from scipy import ndimage


class Processor:
    """
    This class will help with preprocessing your data in PythoStitcher. You don't need
    to execute this script yourself, PythoStitcher will automatically perform the
    preprocessing. If this script throws any error, double check that your data:
        - is in .tif or .mrxs format
        - has multiple resolution layers (pyramidal)
    """
    def __init__(self, image_file, mask_file, save_dir, level, count):

        assert isinstance(level, int), "level must be an integer"

        self.image_filename = image_file
        self.mask_filename = mask_file
        self.mask_provided = bool(mask_file)
        self.save_dir = save_dir
        self.new_level = level
        self.count = count

        self.opener = mir.MultiResolutionImageReader()

        return

    def load(self):
        """
        Function to load and downsample the raw mask to the provided level. If no mask
        is provided, it will be created through some simple processing.
        """

        # load raw image
        self.raw_image = self.opener.open(str(self.image_filename))
        assert self.raw_image.valid(), "Loaded image was not valid"

        # Load raw mask if available
        if self.mask_provided:
            self.raw_mask = self.opener.open(str(self.mask_filename))
            assert self.raw_mask.valid(), "Loaded mask was not valid"

        # Get downsampled image
        self.new_dims = self.raw_image.getLevelDimensions(self.new_level)
        self.image = self.raw_image.getUCharPatch(0, 0, *self.new_dims, self.new_level)

        # Get downsampled mask with same dimensions as downsampled image
        if self.mask_provided:
            mask_dims = [
                self.raw_mask.getLevelDimensions(i)
                for i in range(self.raw_mask.getNumberOfLevels())
            ]
            mask_level = mask_dims.index(self.new_dims)
            self.mask = self.raw_mask.getUCharPatch(0, 0, *self.new_dims, mask_level)
            self.mask = np.squeeze(self.mask)
            self.mask = ((self.mask / np.max(self.mask)) * 255).astype("uint8")

        else:
            raise ValueError("PythoStitcher requires a tissue mask for stitching")

        return

    def get_otsu_mask(self):
        """
        Method to get the mask using Otsu thresholding. This mask will be combined with
        the tissue segmentation mask in order to filter out the fatty tissue.
        """

        # Convert to HSV space and perform Otsu thresholding
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        self.image_hsv = cv2.medianBlur(self.image_hsv[:, :, 1], 7)
        _, self.otsu_mask = cv2.threshold(self.image_hsv, 0, 255, cv2.THRESH_OTSU +
                                          cv2.THRESH_BINARY)
        self.otsu_mask = (self.otsu_mask / np.max(self.otsu_mask)).astype("uint8")

        # Postprocess the mask a bit
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(8, 8))
        pad = int(0.1 * self.otsu_mask.shape[0])
        self.otsu_mask = np.pad(
            self.otsu_mask,
            [[pad, pad], [pad, pad]],
            mode="constant",
            constant_values=0
        )
        self.otsu_mask = cv2.morphologyEx(
            src=self.otsu_mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=3
        )
        self.otsu_mask = self.otsu_mask[pad:-pad, pad:-pad]

        return

    def get_tissueseg_mask(self):
        """
        Function to postprocess both the regular and the mask. This mainly consists
        of getting the largest component and then cleaning up this mask.
        """

        # Get information on all connected components in the mask
        num_labels, labeled_im, stats, _ = cv2.connectedComponentsWithStats(
            self.mask, connectivity=8
        )

        # Background gets counted as label, therefore an empty image will have 1 label.
        assert num_labels > 1, "mask is empty"

        # The largest label (excluding background) is the mask.
        largest_cc_label = np.argmax(stats[1:, -1]) + 1
        self.mask = ((labeled_im == largest_cc_label) * 255).astype("uint8")

        # Temporarily enlarge mask for better closing
        pad = int(0.1 * self.mask.shape[0])
        self.mask = np.pad(
            self.mask,
            [[pad, pad], [pad, pad]],
            mode="constant",
            constant_values=0
        )

        # Closing operation to close some holes on the mask border
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))
        self.mask = cv2.morphologyEx(src=self.mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=2)

        # Flood fill to remove holes inside mask. The floodfill mask is required by opencv
        seedpoint = (0, 0)
        floodfill_mask = np.zeros((self.mask.shape[0] + 2, self.mask.shape[1] + 2)).astype("uint8")
        _, _, self.mask, _ = cv2.floodFill(self.mask, floodfill_mask, seedpoint, 255)
        self.mask = self.mask[1+pad:-1-pad, 1+pad:-1-pad]
        self.mask = 1 - self.mask

        assert np.sum(self.mask) > 0, "floodfilled mask is empty"

        return

    def combine_masks(self):
        """
        Method to combine the tissue mask and the Otsu mask for fat filtering.
        """

        # Combine
        self.final_mask = self.otsu_mask * self.mask

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
        floodfill_mask = np.zeros((self.final_mask.shape[0] + 2, self.final_mask.shape[1] + 2)).astype("uint8")
        _, _, self.final_mask, _ = cv2.floodFill(self.final_mask, floodfill_mask, seedpoint, 255)
        self.final_mask = self.final_mask[1 + offset:-1 - offset, 1 + offset:-1 - offset]
        self.final_mask = 1 - self.final_mask

        # Crop to nonzero pixels for efficient saving
        r, c = np.nonzero(self.final_mask)
        self.final_mask = self.final_mask[np.min(r) : np.max(r), np.min(c) : np.max(c)]
        self.image = self.image[np.min(r) : np.max(r), np.min(c) : np.max(c)]
        self.image = self.image.astype("uint8")
        self.final_mask = (self.final_mask * 255).astype("uint8")

        return

    def normalize_stains(self):
        """
        Function to normalize the stain of the image.
        """
        
        # Only normalize stains for the other images
        if self.count > 1:
            
            # Load reference image
            ref_image = cv2.imread(str(self.save_dir.joinpath("preprocessed_images", f"fragment1.png")), cv2.IMREAD_COLOR)
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            
            # Initialize stain normalizer
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x*255)
            ])
            stain_normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch")
            stain_normalizer.fit(T(ref_image))
            
            # Apply stain normalization
            self.image = stain_normalizer.normalize(T(self.image))
            self.image = self.image.numpy().astype("uint8")
        
        return

    def save(self):
        """
        Function to save the downsampled image and mask
        """

        # Save image
        image_savedir = self.save_dir.joinpath("preprocessed_images")
        if not image_savedir.is_dir():
            image_savedir.mkdir()

        image_savefile = image_savedir.joinpath(f"fragment{self.count}.png")
        cv2.imwrite(str(image_savefile), cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

        # Save mask
        mask_savedir = self.save_dir.joinpath("preprocessed_masks")
        if not mask_savedir.is_dir():
            mask_savedir.mkdir()

        mask_savefile = mask_savedir.joinpath(f"fragment{self.count}.png")
        cv2.imwrite(str(mask_savefile), self.final_mask)

        return


def prepare_data(parameters):
    """
    Downsample both images and masks to determine fragment configuration.
    """

    parameters["log"].log(parameters["my_level"], "Preprocessing raw images...")

    # Get all image files
    image_files = sorted(
        [i for i in parameters["data_dir"].joinpath("raw_images").iterdir() if not i.is_dir()]
    )

    # Get mask files if these are provided
    masks_provided = parameters["data_dir"].joinpath("raw_masks").is_dir()
    if masks_provided:
        mask_files = sorted([i for i in parameters["data_dir"].joinpath("raw_masks").iterdir()])
        assert len(image_files) == len(mask_files), "found unequal number of image/mask files!"
    else:
        mask_files = [None] * len(image_files)

    # Process and save image with corresponding mask (if available)
    for c, vars in enumerate(zip(image_files, mask_files), 1):
        image, mask = vars
        parameters["log"].log(parameters["my_level"], f" - {image.name.split('.')[0]}")
        data_processor = Processor(
            image_file=image,
            mask_file=mask,
            save_dir=parameters["save_dir"],
            level=parameters["image_level"],
            count=c,
        )
        data_processor.load()
        data_processor.get_otsu_mask()
        data_processor.get_tissueseg_mask()
        data_processor.combine_masks()
        data_processor.normalize_stains()
        data_processor.save()

    parameters["log"].log(parameters["my_level"], " > finished!\n")

    return
