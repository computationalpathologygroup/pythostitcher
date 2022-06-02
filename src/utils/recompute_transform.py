import numpy as np
import math
from skimage.transform import EuclideanTransform, matrix_transform


def recompute_transform(quadrant_A, quadrant_B, quadrant_C, quadrant_D, tform):
    """
    Custom function to recompute a transformation matrix.

    The rotation function by skimage rotates the image around the point (0, 0), thereby inducing an additional
    translation component in the rotation. To counteract this translation, we compute the center point of the
    bbox before and after rotation. This essentially results in a rotation of the image around its center,
    thereby negating the additional translation component. This is also performed in the initial alignment step.
    """

    # Get transformation per quadrant from genetic algorithm tform
    ga_tform_A = tform[:3]
    ga_tform_B = tform[3:6]
    ga_tform_C = tform[6:9]
    ga_tform_D = tform[9:]

    # Compute center of mass of bbox
    center_a_pre = np.mean(quadrant_A.bbox_corners, axis=0)
    center_b_pre = np.mean(quadrant_B.bbox_corners, axis=0)
    center_c_pre = np.mean(quadrant_C.bbox_corners, axis=0)
    center_d_pre = np.mean(quadrant_D.bbox_corners, axis=0)

    # Create transformation with only rotation
    rot_tform_a = EuclideanTransform(rotation=-math.radians(ga_tform_A[2]), translation=(0, 0))
    rot_tform_b = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=(0, 0))
    rot_tform_c = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=(0, 0))
    rot_tform_d = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=(0, 0))

    # Rotate the bbox corners
    rot_bbox_corners_a = matrix_transform(quadrant_A.bbox_corners, rot_tform_a.params)
    rot_bbox_corners_b = matrix_transform(quadrant_B.bbox_corners, rot_tform_b.params)
    rot_bbox_corners_c = matrix_transform(quadrant_C.bbox_corners, rot_tform_c.params)
    rot_bbox_corners_d = matrix_transform(quadrant_D.bbox_corners, rot_tform_d.params)

    # Compute the new center of mass of the bbox
    center_a_post = np.mean(rot_bbox_corners_a, axis=0)
    center_b_post = np.mean(rot_bbox_corners_b, axis=0)
    center_c_post = np.mean(rot_bbox_corners_c, axis=0)
    center_d_post = np.mean(rot_bbox_corners_d, axis=0)

    # The additional translation is approximately the difference in the COM location
    trans_a = np.round(center_a_post - center_a_pre)
    trans_b = np.round(center_b_post - center_b_pre)
    trans_c = np.round(center_c_post - center_c_pre)
    trans_d = np.round(center_d_post - center_d_pre)

    # Include this translation in the original transformation
    final_tform_a = ga_tform_A - (*trans_a, 0)
    final_tform_b = ga_tform_B - (*trans_b, 0)
    final_tform_c = ga_tform_C - (*trans_c, 0)
    final_tform_d = ga_tform_D - (*trans_d, 0)

    return final_tform_a, final_tform_b, final_tform_c, final_tform_d
