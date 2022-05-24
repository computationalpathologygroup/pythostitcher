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
    bbox_corners_a = [quadrant_A.bbox_corner_a, quadrant_A.bbox_corner_b,
                      quadrant_A.bbox_corner_c, quadrant_A.bbox_corner_d]
    bbox_corners_b = [quadrant_B.bbox_corner_a, quadrant_B.bbox_corner_b,
                      quadrant_B.bbox_corner_c, quadrant_B.bbox_corner_d]
    bbox_corners_c = [quadrant_C.bbox_corner_a, quadrant_C.bbox_corner_b,
                      quadrant_C.bbox_corner_c, quadrant_C.bbox_corner_d]
    bbox_corners_d = [quadrant_D.bbox_corner_a, quadrant_D.bbox_corner_b,
                      quadrant_D.bbox_corner_c, quadrant_D.bbox_corner_d]
    center_a_pre = np.mean(bbox_corners_a, axis=0)
    center_b_pre = np.mean(bbox_corners_b, axis=0)
    center_c_pre = np.mean(bbox_corners_c, axis=0)
    center_d_pre = np.mean(bbox_corners_d, axis=0)

    # Create transformation with only rotation
    rot_tform_a = EuclideanTransform(rotation=-math.radians(ga_tform_A[2]), translation=(0, 0))
    rot_tform_b = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=(0, 0))
    rot_tform_c = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=(0, 0))
    rot_tform_d = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=(0, 0))

    # Rotate the bbox corners
    rot_bbox_corners_a = np.squeeze(matrix_transform(bbox_corners_a, rot_tform_a.params))
    rot_bbox_corners_b = np.squeeze(matrix_transform(bbox_corners_b, rot_tform_b.params))
    rot_bbox_corners_c = np.squeeze(matrix_transform(bbox_corners_c, rot_tform_c.params))
    rot_bbox_corners_d = np.squeeze(matrix_transform(bbox_corners_d, rot_tform_d.params))

    # # Compute the new center of mass of the bbox
    center_a_post = np.mean(rot_bbox_corners_a, axis=0)
    center_b_post = np.mean(rot_bbox_corners_b, axis=0)
    center_c_post = np.mean(rot_bbox_corners_c, axis=0)
    center_d_post = np.mean(rot_bbox_corners_d, axis=0)

    # The additional translation is approximately the difference in the COM location
    trans_a = center_a_pre - center_a_post
    trans_b = center_b_pre - center_b_post
    trans_c = center_c_pre - center_c_post
    trans_d = center_d_pre - center_d_post

    # Include this translation in the original transformation
    final_tform_a = ga_tform_A + [*trans_a, 0]
    final_tform_b = ga_tform_B + [*trans_b, 0]
    final_tform_c = ga_tform_C + [*trans_c, 0]
    final_tform_d = ga_tform_D + [*trans_d, 0]

    return final_tform_a, final_tform_b, final_tform_c, final_tform_d