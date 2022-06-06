import numpy as np

from .transformations import *


def recompute_transform(quadrant_A, quadrant_B, quadrant_C, quadrant_D, tform):
    """
    Custom function to recompute a transformation matrix.

    The rotation function by skimage rotates the image around the point (0, 0), thereby inducing an additional
    translation component in the rotation. To counteract this translation, we compute the center point of the
    bbox before and after rotation. This essentially results in a rotation of the image around its center,
    thereby negating the additional translation component. This is also performed in the initial alignment step.
    """

    tform_A = tform[quadrant_A.quadrant_name]
    tform_B = tform[quadrant_B.quadrant_name]
    tform_C = tform[quadrant_C.quadrant_name]
    tform_D = tform[quadrant_D.quadrant_name]

    # Compute center of mass of bbox
    center_a_pre = np.mean(quadrant_A.bbox_corners2, axis=0)
    center_b_pre = np.mean(quadrant_B.bbox_corners2, axis=0)
    center_c_pre = np.mean(quadrant_C.bbox_corners2, axis=0)
    center_d_pre = np.mean(quadrant_D.bbox_corners2, axis=0)

    # Rotate the bbox corners
    rot_bbox_corners_a = warp_2d_points(src=quadrant_A.bbox_corners2, rotation=quadrant_A.angle2, translation=(0, 0))
    rot_bbox_corners_b = warp_2d_points(src=quadrant_B.bbox_corners2, rotation=quadrant_B.angle2, translation=(0, 0))
    rot_bbox_corners_c = warp_2d_points(src=quadrant_C.bbox_corners2, rotation=quadrant_C.angle2, translation=(0, 0))
    rot_bbox_corners_d = warp_2d_points(src=quadrant_D.bbox_corners2, rotation=quadrant_D.angle2, translation=(0, 0))

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
    final_tform_a = np.array([tform_A[0]-trans_a[0], tform_A[1]-trans_a[1], tform_A[2], tform_A[3]], dtype=object)
    final_tform_b = np.array([tform_B[0]-trans_b[0], tform_B[1]-trans_b[1], tform_B[2], tform_B[3]], dtype=object)
    final_tform_c = np.array([tform_C[0]-trans_c[0], tform_C[1]-trans_c[1], tform_C[2], tform_C[3]], dtype=object)
    final_tform_d = np.array([tform_D[0]-trans_d[0], tform_D[1]-trans_d[1], tform_D[2], tform_D[3]], dtype=object)

    return final_tform_a, final_tform_b, final_tform_c, final_tform_d


def merge_transformations(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, initial_tform, ga_dict):
    """
    Custom function to merge the initial transformation and the transformation acquired by the genetic algorithm.

    NOTE: this function merges the transformations WITHOUT accounting for the translation induced due to rotation.
    Omitting this extra translation is necessary for upscaling the transformation for the next resolution.
    """

    # Get transformation in X
    tx_A = np.round(initial_tform[quadrant_A.quadrant_name][0] + ga_dict[quadrant_A.quadrant_name][0])
    tx_B = np.round(initial_tform[quadrant_B.quadrant_name][0] + ga_dict[quadrant_B.quadrant_name][0])
    tx_C = np.round(initial_tform[quadrant_C.quadrant_name][0] + ga_dict[quadrant_C.quadrant_name][0])
    tx_D = np.round(initial_tform[quadrant_D.quadrant_name][0] + ga_dict[quadrant_D.quadrant_name][0])
    all_tx = [tx_A, tx_B, tx_C, tx_D]

    # Get transformation in Y
    ty_A = np.round(initial_tform[quadrant_A.quadrant_name][1] + ga_dict[quadrant_A.quadrant_name][1])
    ty_B = np.round(initial_tform[quadrant_B.quadrant_name][1] + ga_dict[quadrant_B.quadrant_name][1])
    ty_C = np.round(initial_tform[quadrant_C.quadrant_name][1] + ga_dict[quadrant_C.quadrant_name][1])
    ty_D = np.round(initial_tform[quadrant_D.quadrant_name][1] + ga_dict[quadrant_D.quadrant_name][1])
    all_ty = [ty_A, ty_B, ty_C, ty_D]

    # Combine the angle obtained from both transformations
    angle_A = np.round(initial_tform[quadrant_A.quadrant_name][2] + ga_dict[quadrant_A.quadrant_name][2], 1)
    angle_B = np.round(initial_tform[quadrant_B.quadrant_name][2] + ga_dict[quadrant_B.quadrant_name][2], 1)
    angle_C = np.round(initial_tform[quadrant_C.quadrant_name][2] + ga_dict[quadrant_C.quadrant_name][2], 1)
    angle_D = np.round(initial_tform[quadrant_D.quadrant_name][2] + ga_dict[quadrant_D.quadrant_name][2], 1)
    all_angles = [angle_A, angle_B, angle_C, angle_D]

    merged_tform = dict()
    for key, tx, ty, a in zip(parameters["filenames"].keys(), all_tx, all_ty, all_angles):
        merged_tform[key] = [tx, ty, a, parameters["output_shape"]]

    return merged_tform
