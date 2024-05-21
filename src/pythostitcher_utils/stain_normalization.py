import torchstain
import numpy as np
import pyvips
from typing import Tuple
from torchvision import transforms


class Reinhard_normalizer(object):
    """
    A stain normalization object for PyVips. Fits a reference PyVips image,
    transforms a PyVips Image. Can also be initialized with precalculated
    means and stds (in LAB colorspace).

    Adapted from https://gist.github.com/munick/badb6582686762bb10265f8a66c26d48
    """

    def __init__(self, target_means=None, target_stds=None):
        self.target_means = target_means
        self.target_stds  = target_stds

        return

    def fit(self, target: pyvips.Image): 
        """
        Fit a Pyvips image.
        """

        # Get the means and stds of the target image
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds  = stds

        return
    
    def transform(self, image):
        """
        Method to apply the transformation to a PyVips image.
        """
        
        # Split the image into LAB channels
        L, A, B = self.lab_split(image)
        means, stds = self.get_mean_std(image)

        # Apply normalization to each channel
        norm1 = ((L - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((A - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((B - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]

        return self.merge_to_rgb(norm1, norm2, norm3)
    
    def lab_split(self, img: pyvips.Image) -> Tuple[pyvips.Image, pyvips.Image, pyvips.Image]:
        """
        Method to convert a PyVips image to LAB colorspace.
        """

        img_lab = img.colourspace("VIPS_INTERPRETATION_LAB")
        L, A, B = img_lab.bandsplit()[:3]

        return L, A, B
        
    def get_mean_std(self, image: pyvips.Image) -> Tuple:
        """
        Method to calculate the mean and standard deviation of a PyVips image.
        """

        L, A, B = self.lab_split(image)
        m1, sd1 = L.avg(), L.deviate()
        m2, sd2 = A.avg(), A.deviate()
        m3, sd3 = B.avg(), B.deviate()
        means = m1, m2, m3
        stds  = sd1, sd2, sd3
        self.image_stats = means, stds

        return means, stds
    
    def merge_to_rgb(self, L: pyvips.Image, A: pyvips.Image, B: pyvips.Image) -> pyvips.Image:
        """
        Method to merge the L, A, B bands to an RGB image.
        """

        img_lab = L.bandjoin([A,B])
        img_rgb = img_lab.colourspace('VIPS_INTERPRETATION_sRGB')

        return img_rgb


def apply_stain_norm(images):
    """
    Function to apply stain normalization to a set of regular images.
    """

    # Always use first image as reference
    ref_image = images[0]

    # Initiate normalizer
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    stain_normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch")
    stain_normalizer.fit(T(ref_image))

    normalized_images = []

    # Apply stain normalization
    for image in images:
        norm_im = stain_normalizer.normalize(T(image))
        norm_im = norm_im.numpy().astype("uint8")
        normalized_images.append(norm_im)

    return normalized_images


def apply_fullres_stain_norm(images):
    """
    Function to apply stain normalization on full resolution pyvips images.
    """

    # Always use first image as reference
    ref_image = images[0]

    normalizer = Reinhard_normalizer()
    normalizer.fit(ref_image)

    normalized_images = []

    # Apply stain normalization
    for image in images:
        norm_im = normalizer.transform(image)
        normalized_images.append(norm_im)

    return normalized_images
