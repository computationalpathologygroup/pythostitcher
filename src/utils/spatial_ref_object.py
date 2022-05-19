import numpy as np


class spatial_ref_object:
    """
    Python implementation for the imref2d class in Matlab
    """

    def __init__(self, image):

        assert len(np.shape(image)) == 2, "image must be 2-dimensional"

        # Image shape
        self.image_size = image.shape

        # Internal world limits. Equals regular world limits.
        self.xworld_lim_internal = [0.5, self.image_size[0]+0.5]
        self.yworld_lim_internal = [0.5, self.image_size[1]+0.5]
        self.xworld_lim = self.xworld_lim_internal
        self.yworld_lim = self.yworld_lim_internal

        # Image extent in world
        self.im_extent_worldx = np.diff(self.xworld_lim)
        self.im_extent_worldy = np.diff(self.yworld_lim)
        self.pix_extent_worldx = np.diff(self.xworld_lim)/self.image_size[1]
        self.pix_extent_worldy = np.diff(self.yworld_lim)/self.image_size[0]

        # Intrinsic limits
        self.intrinsic_limx = [0.5, self.image_size[1] + 0.5]
        self.intrinsic_limy = [0.5, self.image_size[0] + 0.5]

        # Recompute all transforms
        self.recompute_transforms()

        return


    def intrinsic_to_world(self, x_intrinsic, y_intrinsic):
        """
        Converts intrinsic coordinates to world coordinates
        """
        M = self.tform_intrinsic_world
        xw = M[0, 0] * x_intrinsic + M[2, 0]
        yw = M[1, 1] * y_intrinsic + M[2, 1]

        return xw, yw


    def world_to_intrinsic(self, x_world, y_world):
        """
        Converts world coordinates to intrinsic coordinates
        """
        M = self.tform_world_intrinsic
        xi = M[0][0] * x_world + M[2][0]
        yi = M[1][1] * y_world + M[2][1]

        return xi, yi


    def world_to_subscript(self, x_world, y_world):
        """
        Convert world coordinates to subscripts
        """

        TF = compare_limits(self, x_world, y_world)
        c, r = world_to_intrinsic(self, x_world, y_world)

        r[TF] = np.max(1, np.min(np.round(r[TF]), self.image_size[0]))
        c[TF] = np.max(1, np.min(np.round(c[TF]), self.image_size[1]))

        r[~TF] = np.nan
        c[~TF] = np.nan

        return r, c


    def compare_limits(self, x_world, y_world):
        """
        Compare values of x_world and y_world with world limits
        """

        c1 = x_world >= self.xworld_lim[0]
        c2 = x_world <= self.xworld_lim[1]
        c3 = y_world >= self.yworld_lim[0]
        c4 = y_world <= self.yworld_lim[1]
        result = (c1 and c2 and c3 and c4)

        return result


    def sizes_match(self, im):
        """
        Function to check whether the shape of an input equals the internal shape in the ref object
        """

        im_size = np.shape(im)

        c1 = im_size[0] == self.image_size[0]
        c2 = im_size[1] == self.image_size[1]

        result = (c1 and c2)

        return result

    def recompute_transforms(self):

        sx = self.pix_extent_worldx
        sy = self.pix_extent_worldy
        tx = self.xworld_lim[0]
        ty = self.yworld_lim[0]
        shift = np.array([[1, 0, 0], [0, 1, 0], [-0.5, -0.5, 1]], dtype=object)

        self.tform_intrinsic_world = shift * np.array([[sx, 0, 0], [0, sy, 0], [tx, ty, 1]], dtype=object)

        sx = 1/self.pix_extent_worldx
        sy = 1/self.pix_extent_worldy
        tx = self.intrinsic_limx[0] - 1/self.pix_extent_worldx * self.xworld_lim[0]
        ty = self.intrinsic_limy[0] - 1/self.pix_extent_worldy * self.yworld_lim[0]
        self.tform_world_intrinsic = [[sx, 0, 0], [0, sy, 0], [tx, ty, 1]]

        return
