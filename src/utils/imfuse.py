import numpy as np
from skimage.color import rgb2gray
from scipy.interpolate import griddata

from .spatial_ref_object import spatial_ref_object

# The original script also called an argument parser as function. However, this was rather intricately
# designed and cannot be easily replicated in Python. As long as suitable input arguments are given,
# omission of this function will not be detrimental, but this should still be noted.


class imfuse:
    def __init__(self, A, B, RA, RB, method, options):
        self.A = A
        self.B = B
        self.RA = RA
        self.RB = RB
        self.method = method
        self.options = options

        if "ColorChannels" in self.options.keys():
            if self.options["ColorChannels"] == "green-magenta":
                self.options["ColorChannels"] = [2, 1, 2]
            elif self.options["ColorChannels"] == "red-cyan":
                self.options["ColorChannels"] = [1, 2, 2]

        self.AisRGB = len(np.shape(np.squeeze(self.A))) == 3
        self.BisRGB = len(np.shape(np.squeeze(self.B))) == 3

        self.A, self.B, self.A_mask, self.B_mask, self.RC = calculateOverlayImages(self.A, self.RA, self.B, self.RB)

        if self.method == "blend":
            self.C = local_create_blend
        elif self.method == "falsecolor":
            self.C = local_create_RGB(options["ColorChannels"])
        elif self.method == "diff":
            self.C = local_createDiff
        elif self.method == "montage":
            self.C = local_createSideBySide
            self.RC = np.eye(3)

        return

    def local_create_blend(self):

        self.A, self.B = makeSimilar(self.A, self.B, self.AisRGB, self.BisRGB, self.scaling)

        onlyA = ((self.A_mask == 1) & (self.B_mask == 0))
        onlyB = ((self.A_mask == 0) & (self.B_mask == 1))
        bothAandB = ((self.A_mask == 1) & (self.B_mask == 1))

        weight1, weight2 = 0.5, 0.5

        self.result = np.zeros((np.shape(self.A)))

        for i in range(np.shape(self.A)[-2]):
            a = self.A[:, :, i]
            b = self.B[:, :, i]
            r = self.result[:, :, i]
            r[onlyA] = a[onlyA]
            r[onlyB] = b[onlyB]

            bothAandBA0 = bothAandB * b
            bothAandBB0 = bothAandB * a

            r[bothAandBA0] = b[bothBandBA0]
            r[bothAandBB0] = a[bothAandBB0]

            r[self.objectoverlap] = weight1 * a[self.objectoverlap] + weight2 * b[self.objectoverlap]

            self.result[:, :, i] = r

        return self.result


    def local_create_RGB(self, channels=(2, 1, 2)):

        if np.shape(self.A)[2] > 1:
            self.A = rgb2gray(self.A)

        if np.shape(self.B)[2] > 1:
            self.B = rgb2gray(self.B)

        if self.scaling == "none":
            pass
        elif self.scaling == "joint":
            self.A, self.B = scaleTwoGrayscaleImages(self.A, self.B)
        elif self.scaling == "independent":
            self.A = scaleGrayscaleImage(self.A)
            self.B = scaleGrayscaleImage(self.B)

        self.result = np.zeros((np.shape(self.A)[0], np.shape(self.A)[1], 3))

        for i in range(3):
            if channels[i] == 1:
                self.result[:, :, i] = self.A
            elif channels[i] == 2:
                self.result[:, :, i] = self.B

        return self.result

    def local_createDiff(self):

        if np.shape(self.A[2]) > 1:
            self.A = rgb2gray(self.A)
        if np.shape(self.B[2]) > 1:
            self.B = rgb2gray(self.B)

        if self.scaling == "none":
            pass
        elif self.scaling == "joint":
            self.A, self.B = scaleTwoGrayscaleImages(self.A, self.B)
        elif self.scaling == "independent":
            self.A = scaleGrayscaleImage(self.A)
            self.B = scaleGrayscaleImage(self.B)

        self.result = scaleGrayscaleImage(np.abs(self.A-self.B))

        return self.result

    def local_createSideBySide(self):

        self.A, self.B = makeSimilar(self.A, self.B, self.AisRGB, self.BisRGB, self.scaling)
        self.result = [self.A, self.B]

        return self.result


def makeSimilar(A, B, AisRGB, BisRGB, scaling):

    if (not AisRGB) and (not BisRGB):
        if scaling == "none":
            pass
        elif scaling == "joint":
            A, B = scaleTwoGrayscaleImages(A, B)
        elif scaling == "independent":
            A = scaleGrayscaleImage(A)
            B = scaleGrayscaleImage(B)

    elif (AisRGB and BisRGB):
        pass

    elif (AisRGB) and (not BisRGB):
        if scaling == "none":
            B = np.tile(B[:, :, np.newaxis], 3)
        else:
            B = scaleGrayscaleImage(B)
            B = np.tile(B[:, :, np.newaxis], 3)

    elif (not AisRGB) and (BisRGB):
        if scaling == "none":
            A = np.tile(A[:, :, np.newaxis], 3)
        else:
            A = scaleGrayscaleImage(A)
            A = np.tile(A[:, :, np.newaxis], 3)

    return A, B


def scaleGrayscaleImage(image_data):

    image_data = image_data - np.min(image_data)

    image_data = image_data / np.max(image_data)

    return image_data


def scaleTwoGrayscaleImages(A, B):

    image_data = [A, B]

    image_data = scaleGrayscaleImage(image_data)

    A = image_data[:, :np.shape(A)[1], :]
    B = image_data[:, np.shape(A)[1]:, :]

    return A, B


def resampleImageToNewSpatialRef(A, RA, RB, method, fill_value):
    """
    Helper function based on matlab function with same name
    """

    assert len(np.shape(A)) == 2, f"input must be 2D, received {len(np.shape(A))}"

    # Make meshgrid from spatial ref object of B
    b_intrinsic_x, b_intrinsic_y = np.meshgrid(np.arange(0, RB.image_size[1]), np.arange(0, RB.image_size[0]))

    # Convert to world coordinates B
    worldoverlay_x, worldoverlay_y = RB.intrinsic_to_world(b_intrinsic_x, b_intrinsic_y)

    # Convert to intrinsic coordinates A
    x_intrinsic, y_intrinsic = RA.world_to_intrinsic(worldoverlay_x, worldoverlay_y)

    # Perform interpolation
    meshA = np.meshgrid(np.arange(0, np.shape(A)[0]), np.arange(0, np.shape(A)[1]))
    xi = np.array([[x, y] for x, y in zip(np.ravel(x_intrinsic), np.ravel(y_intrinsic))])
    B = griddata(meshA, A, xi, method=method, fill_value=fill_value)

    return B


def calculateOverlayImages(A, RA, B, RB):

    outputWorldLimitsX = [np.min([RA.xworld_lim[0], RB.xworld_lim[0]]),
                          np.max([RA.xworld_lim[1], RB.xworld_lim[1]])]

    outputWorldLimitsY = [np.min([RA.yworld_lim[0], RB.yworld_lim[0]]),
                          np.max([RA.yworld_lim[1], RB.yworld_lim[1]])]

    goalResolutionX = np.min([RA.pix_extent_worldx, RB.pix_extent_worldx])
    goalResolutionY = np.min([RA.pix_extent_worldy, RB.pix_extent_worldy])

    widthOutputRaster = np.ceil(np.diff(outputWorldLimitsX) / goalResolutionX)
    heightOutputRaster = np.ceil(np.diff(outputWorldLimitsY) / goalResolutionY)

    R_output = spatial_ref_object(np.zeros((int(heightOutputRaster), int(widthOutputRaster))))
    R_output.xworld_lim = outputWorldLimitsX
    R_output.yworld_lim = outputWorldLimitsY

    fillVal = 0
    #A_padded = resampleImageToNewSpatialRef(A, RA, R_output, method='linear', fill_value=fillVal)
    #B_padded = resampleImageToNewSpatialRef(B, RB, R_output, method='linear', fill_value=fillVal)

    pad_shape = [R_output.image_size[0] - np.shape(A)[0], R_output.image_size[1] - np.shape(A)[1]]
    A_padded = np.pad(A, pad_shape)
    pad_shape = [R_output.image_size[0] - np.shape(B)[0], R_output.image_size[1] - np.shape(B)[1]]
    B_padded = np.pad(B, pad_shape)

    outputIntrinsicX, outputIntrinsicY = np.meshgrid(np.arange(0, R_output.image_size[1]), np.arange(0, R_output.image_size[0]))
    xWorldOverlayLoc, yWorldOverlayLoc = R_output.intrinsic_to_world(outputIntrinsicX, outputIntrinsicY)
    A_mask = ((xWorldOverlayLoc in RA) or (yWorldOverlayLoc in RA))
    B_mask = ((xWorldOverlayLoc in RB) or (yWorldOverlayLoc in RB))

    return A_padded, B_padded, A_mask, B_mask, R_output

