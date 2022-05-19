import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle
from skimage.transform import EuclideanTransform, matrix_transform, warp


def plot_rotation_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D):
    """
    Custom function to plot the result of the automatic rotation
    """

    # Pad the image for pretty plotting
    pad = int(np.round(0.25 * np.shape(quadrant_A.mask)[0]))

    # Get x/y values of bbox
    A_corners_x = [c[0]+pad for c in quadrant_A.bbox_corners]
    A_corners_y = [c[1]+pad for c in quadrant_A.bbox_corners]
    B_corners_x = [c[0]+pad for c in quadrant_B.bbox_corners]
    B_corners_y = [c[1]+pad for c in quadrant_B.bbox_corners]
    C_corners_x = [c[0]+pad for c in quadrant_C.bbox_corners]
    C_corners_y = [c[1]+pad for c in quadrant_C.bbox_corners]
    D_corners_x = [c[0]+pad for c in quadrant_D.bbox_corners]
    D_corners_y = [c[1]+pad for c in quadrant_D.bbox_corners]

    # Add first value again to close the box
    A_corners_x.append(A_corners_x[0])
    A_corners_y.append(A_corners_y[0])
    B_corners_x.append(B_corners_x[0])
    B_corners_y.append(B_corners_y[0])
    C_corners_x.append(C_corners_x[0])
    C_corners_y.append(C_corners_y[0])
    D_corners_x.append(D_corners_x[0])
    D_corners_y.append(D_corners_y[0])

    ## Plot quadrant A
    plt.figure(figsize=(6, 14))
    plt.suptitle(f"Quadrants before and after automatic rotation", fontsize=16)
    plt.subplot(421)
    plt.axis("off")
    plt.title(quadrant_A.quadrant_name)
    plt.imshow(np.pad(quadrant_A.mask, [pad, pad]), cmap="gray")
    plt.scatter(quadrant_A.mask_corner_a[0] + pad, quadrant_A.mask_corner_a[1] + pad, facecolor="r", s=100)
    plt.plot(A_corners_x, A_corners_y, linewidth=4, c="r")
    plt.subplot(422)
    plt.axis("off")
    plt.title(quadrant_A.quadrant_name)
    plt.imshow(np.pad(quadrant_A.rot_mask, [pad, pad]), cmap="gray")

    ## Plot quadrant B
    plt.subplot(423)
    plt.axis("off")
    plt.title(quadrant_B.quadrant_name)
    plt.imshow(np.pad(quadrant_B.mask, [pad, pad]), cmap="gray")
    plt.scatter(quadrant_B.mask_corner_a[0] + pad, quadrant_B.mask_corner_a[1] + pad, facecolor="r", s=100)
    plt.plot(B_corners_x, B_corners_y, linewidth=4, c="r")
    plt.subplot(424)
    plt.axis("off")
    plt.title(quadrant_B.quadrant_name)
    plt.imshow(np.pad(quadrant_B.rot_mask, [pad, pad]), cmap="gray")

    ## Plot quadrant C
    plt.subplot(425)
    plt.axis("off")
    plt.title(quadrant_C.quadrant_name)
    plt.imshow(np.pad(quadrant_C.mask, [pad, pad]), cmap="gray")
    plt.scatter(quadrant_C.mask_corner_a[0] + pad, quadrant_C.mask_corner_a[1] + pad, facecolor="r", s=100)
    plt.plot(C_corners_x, C_corners_y, linewidth=4, c="r")
    plt.subplot(426)
    plt.axis("off")
    plt.title(quadrant_C.quadrant_name)
    plt.imshow(np.pad(quadrant_C.rot_mask, [pad, pad]), cmap="gray")

    ## Plot quadrant D
    plt.subplot(427)
    plt.axis("off")
    plt.title(quadrant_D.quadrant_name)
    plt.imshow(np.pad(quadrant_D.mask, [pad, pad]), cmap="gray")
    plt.scatter(quadrant_D.mask_corner_a[0] + pad, quadrant_D.mask_corner_a[1] + pad, facecolor="r", s=100)
    plt.plot(D_corners_x, D_corners_y, linewidth=4, c="r")
    plt.subplot(428)
    plt.axis("off")
    plt.title(quadrant_D.quadrant_name)
    plt.imshow(np.pad(quadrant_D.rot_mask, [pad, pad]), cmap="gray")
    plt.tight_layout()
    plt.show()

    return


def plot_transformation_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters):
    """
    Custom function to plot the result of the initial transformation to align the pieces
    """

    combined = quadrant_A.tform_image + quadrant_B.tform_image + \
               quadrant_C.tform_image + quadrant_D.tform_image

    plt.figure()
    current_res = parameters['resolutions'][parameters['iteration']]
    plt.title(f"Initial alignment at resolution {current_res}")
    plt.imshow(combined, cmap="gray")
    plt.show()


def plot_theilsen_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters):
    """
    Custom function to plot the result of the Theil-Sen line estimator
    """

    # Combine all images
    combi_image = quadrant_A.tform_image + quadrant_B.tform_image + quadrant_C.tform_image + quadrant_D.tform_image

    # Plot theilsen lines
    plt.figure()
    plt.title(f"Fitted theilsen lines at resolution {parameters['resolutions'][parameters['iteration']]}")
    plt.imshow(combi_image, cmap="gray")
    plt.plot(quadrant_A.v_edge_theilsen_endpoints[:, 0], quadrant_A.v_edge_theilsen_endpoints[:, 1], linewidth=3, color="g")
    plt.plot(quadrant_A.h_edge_theilsen_endpoints[:, 0], quadrant_A.h_edge_theilsen_endpoints[:, 1], linewidth=3, color="b")
    plt.plot(quadrant_B.v_edge_theilsen_endpoints[:, 0], quadrant_B.v_edge_theilsen_endpoints[:, 1], linewidth=3, color="g")
    plt.plot(quadrant_B.h_edge_theilsen_endpoints[:, 0], quadrant_B.h_edge_theilsen_endpoints[:, 1], linewidth=3, color="b")
    plt.plot(quadrant_C.v_edge_theilsen_endpoints[:, 0], quadrant_C.v_edge_theilsen_endpoints[:, 1], linewidth=3, color="g")
    plt.plot(quadrant_C.h_edge_theilsen_endpoints[:, 0], quadrant_C.h_edge_theilsen_endpoints[:, 1], linewidth=3, color="b")
    plt.plot(quadrant_D.v_edge_theilsen_endpoints[:, 0], quadrant_D.v_edge_theilsen_endpoints[:, 1], linewidth=3, color="g")
    plt.plot(quadrant_D.h_edge_theilsen_endpoints[:, 0], quadrant_D.h_edge_theilsen_endpoints[:, 1], linewidth=3, color="b")
    plt.legend(["v edge", "h edge"])
    plt.show()

    return


def plot_rotated_bbox(quadrant_A, quadrant_B, quadrant_C, quadrant_D):

    scat_x = [quadrant_A.bbox_corner_a[0], quadrant_A.bbox_corner_b[0], quadrant_A.bbox_corner_c[0], quadrant_A.bbox_corner_d[0]]
    scat_y = [quadrant_A.bbox_corner_a[1], quadrant_A.bbox_corner_b[1], quadrant_A.bbox_corner_c[1], quadrant_A.bbox_corner_d[1]]

    plt.figure()
    plt.title("Quadrant A")
    plt.imshow(quadrant_A.tform_image, cmap="gray")
    plt.scatter(scat_x, scat_y, s=25, c="r")
    plt.show()

    scat_x = [quadrant_B.bbox_corner_a[0], quadrant_B.bbox_corner_b[0], quadrant_B.bbox_corner_c[0], quadrant_B.bbox_corner_d[0]]
    scat_y = [quadrant_B.bbox_corner_a[1], quadrant_B.bbox_corner_b[1], quadrant_B.bbox_corner_c[1], quadrant_B.bbox_corner_d[1]]

    plt.figure()
    plt.title("Quadrant B")
    plt.imshow(quadrant_B.tform_image, cmap="gray")
    plt.scatter(scat_x, scat_y, s=25, c="r")
    plt.show()

    scat_x = [quadrant_C.bbox_corner_a[0], quadrant_C.bbox_corner_b[0], quadrant_C.bbox_corner_c[0], quadrant_C.bbox_corner_d[0]]
    scat_y = [quadrant_C.bbox_corner_a[1], quadrant_C.bbox_corner_b[1], quadrant_C.bbox_corner_c[1], quadrant_C.bbox_corner_d[1]]

    plt.figure()
    plt.title("Quadrant C")
    plt.imshow(quadrant_C.tform_image, cmap="gray")
    plt.scatter(scat_x, scat_y, s=25, c="r")
    plt.show()

    scat_x = [quadrant_D.bbox_corner_a[0], quadrant_D.bbox_corner_b[0], quadrant_D.bbox_corner_c[0], quadrant_D.bbox_corner_d[0]]
    scat_y = [quadrant_D.bbox_corner_a[1], quadrant_D.bbox_corner_b[1], quadrant_D.bbox_corner_c[1], quadrant_D.bbox_corner_d[1]]

    plt.figure()
    plt.title("Quadrant D")
    plt.imshow(quadrant_D.tform_image, cmap="gray")
    plt.scatter(scat_x, scat_y, s=25, c="r")
    plt.show()

    return


def plot_tformed_edges(quadrant_A, quadrant_B, quadrant_C, quadrant_D):
    """
    Custom function to plot the transformed edges before inputting them into
    the genetic algorithm
    """

    plt.figure()
    plt.title("Transformed edges")
    plt.plot(quadrant_A.h_edge[:, 0], quadrant_A.h_edge[:, 1], c="b")
    plt.plot(quadrant_A.v_edge[:, 0], quadrant_A.v_edge[:, 1], c="g")
    plt.plot(quadrant_B.h_edge_tform[:, 0], quadrant_B.h_edge_tform[:, 1], c="b")
    plt.plot(quadrant_B.v_edge_tform[:, 0], quadrant_B.v_edge_tform[:, 1], c="g")
    plt.plot(quadrant_C.h_edge_tform[:, 0], quadrant_C.h_edge_tform[:, 1], c="b")
    plt.plot(quadrant_C.v_edge_tform[:, 0], quadrant_C.v_edge_tform[:, 1], c="g")
    plt.plot(quadrant_D.h_edge_tform[:, 0], quadrant_D.h_edge_tform[:, 1], c="b")
    plt.plot(quadrant_D.v_edge_tform[:, 0], quadrant_D.v_edge_tform[:, 1], c="g")
    plt.legend(["Hor", "Ver"])
    plt.show()

    return


def plot_tformed_theilsen_lines(quadrant_A, quadrant_B, quadrant_C, quadrant_D):
    """
    Custom function to plot the transformed Theilsen lines before inputting them into
    the genetic algorithm
    """

    plt.figure()
    plt.title("Transformed Theil-Sen lines")
    plt.plot(quadrant_A.h_edge_theilsen[:, 0], quadrant_A.h_edge_theilsen[:, 1], c="b")
    plt.plot(quadrant_A.v_edge_theilsen[:, 0], quadrant_A.v_edge_theilsen[:, 1], c="g")
    plt.plot(quadrant_B.h_edge_theilsen_tform[:, 0], quadrant_B.h_edge_theilsen_tform[:, 1], c="b")
    plt.plot(quadrant_B.v_edge_theilsen_tform[:, 0], quadrant_B.v_edge_theilsen_tform[:, 1], c="g")
    plt.plot(quadrant_C.h_edge_theilsen_tform[:, 0], quadrant_C.h_edge_theilsen_tform[:, 1], c="b")
    plt.plot(quadrant_C.v_edge_theilsen_tform[:, 0], quadrant_C.v_edge_theilsen_tform[:, 1], c="g")
    plt.plot(quadrant_D.h_edge_theilsen_tform[:, 0], quadrant_D.h_edge_theilsen_tform[:, 1], c="b")
    plt.plot(quadrant_D.v_edge_theilsen_tform[:, 0], quadrant_D.v_edge_theilsen_tform[:, 1], c="g")
    plt.legend(["Hor", "Ver"])
    plt.show()

    return


def plot_tformed_lines(quadrant_A, quadrant_B, quadrant_C, quadrant_D):
    """
    Function to warp the Theil-Sen lines
    """

    # Set tform
    rotation = 5
    rot_angle = -math.radians(rotation)
    tform = EuclideanTransform(rotation=rot_angle, translation=(0, 0))

    # Apply tform
    quadrant_A_ts_h = matrix_transform(quadrant_A.h_edge_theilsen, tform.params)
    quadrant_A_ts_v = matrix_transform(quadrant_A.v_edge_theilsen, tform.params)
    quadrant_B_ts_h = matrix_transform(quadrant_B.h_edge_theilsen, tform.params)
    quadrant_B_ts_v = matrix_transform(quadrant_B.v_edge_theilsen, tform.params)
    quadrant_C_ts_h = matrix_transform(quadrant_C.h_edge_theilsen, tform.params)
    quadrant_C_ts_v = matrix_transform(quadrant_C.v_edge_theilsen, tform.params)
    quadrant_D_ts_h = matrix_transform(quadrant_D.h_edge_theilsen, tform.params)
    quadrant_D_ts_v = matrix_transform(quadrant_D.v_edge_theilsen, tform.params)

    quadrant_A_tform = warp(quadrant_A.tform_image, tform.inverse)
    quadrant_B_tform = warp(quadrant_B.tform_image, tform.inverse)
    quadrant_C_tform = warp(quadrant_C.tform_image, tform.inverse)
    quadrant_D_tform = warp(quadrant_D.tform_image, tform.inverse)

    combi_pre = quadrant_A.tform_image + quadrant_B.tform_image + quadrant_C.tform_image + quadrant_D.tform_image
    combi_post = quadrant_A_tform + quadrant_B_tform + quadrant_C_tform + quadrant_D_tform

    plt.figure()
    plt.subplot(121)
    plt.title("Before extra tform")
    plt.imshow(combi_pre)
    plt.plot(quadrant_A.h_edge_theilsen[:, 0], quadrant_A.h_edge_theilsen[:, 1], c="b")
    plt.plot(quadrant_A.v_edge_theilsen[:, 0], quadrant_A.v_edge_theilsen[:, 1], c="g")
    plt.plot(quadrant_B.h_edge_theilsen[:, 0], quadrant_B.h_edge_theilsen[:, 1], c="b")
    plt.plot(quadrant_B.v_edge_theilsen[:, 0], quadrant_B.v_edge_theilsen[:, 1], c="g")
    plt.plot(quadrant_C.h_edge_theilsen[:, 0], quadrant_C.h_edge_theilsen[:, 1], c="b")
    plt.plot(quadrant_C.v_edge_theilsen[:, 0], quadrant_C.v_edge_theilsen[:, 1], c="g")
    plt.plot(quadrant_D.h_edge_theilsen[:, 0], quadrant_D.h_edge_theilsen[:, 1], c="b")
    plt.plot(quadrant_D.v_edge_theilsen[:, 0], quadrant_D.v_edge_theilsen[:, 1], c="g")

    plt.subplot(122)
    plt.title("After extra tform")
    plt.imshow(combi_post)
    plt.plot(quadrant_A_ts_h[:, 0], quadrant_A_ts_h[:, 1], c="b")
    plt.plot(quadrant_A_ts_v[:, 0], quadrant_A_ts_v[:, 1], c="g")
    plt.plot(quadrant_B_ts_h[:, 0], quadrant_B_ts_h[:, 1], c="b")
    plt.plot(quadrant_B_ts_v[:, 0], quadrant_B_ts_v[:, 1], c="g")
    plt.plot(quadrant_C_ts_h[:, 0], quadrant_C_ts_h[:, 1], c="b")
    plt.plot(quadrant_C_ts_v[:, 0], quadrant_C_ts_v[:, 1], c="g")
    plt.plot(quadrant_D_ts_h[:, 0], quadrant_D_ts_h[:, 1], c="b")
    plt.plot(quadrant_D_ts_v[:, 0], quadrant_D_ts_v[:, 1], c="g")
    plt.show()

    return


def plot_ga_tform(quadrant_A, quadrant_B, quadrant_C, quadrant_D):
    """
    Plotting function to show the transformation of the Theil-Sen lines which was found by the
    genetic algorithm.
    """

    plt.figure()
    plt.subplot(121)
    plt.title("Theil-Sen lines before GA tform")
    plt.plot(quadrant_A.h_edge_theilsen_coords, linewidth=3, color="r")
    plt.plot(quadrant_A.v_edge_theilsen_coords, linewidth=3, color="r")
    plt.plot(quadrant_B.h_edge_theilsen_coords, linewidth=3, color="r")
    plt.plot(quadrant_B.v_edge_theilsen_coords, linewidth=3, color="r")
    plt.plot(quadrant_C.h_edge_theilsen_coords, linewidth=3, color="r")
    plt.plot(quadrant_C.v_edge_theilsen_coords, linewidth=3, color="r")
    plt.plot(quadrant_D.h_edge_theilsen_coords, linewidth=3, color="r")
    plt.plot(quadrant_D.v_edge_theilsen_coords, linewidth=3, color="r")

    plt.subplot(122)
    plt.title("Theil-Sen lines after GA tform")
    plt.plot(quadrant_A.h_edge_theilsen_tform, linewidth=3, color="g")
    plt.plot(quadrant_A.v_edge_theilsen_tform, linewidth=3, color="g")
    plt.plot(quadrant_B.h_edge_theilsen_tform, linewidth=3, color="g")
    plt.plot(quadrant_B.v_edge_theilsen_tform, linewidth=3, color="g")
    plt.plot(quadrant_C.h_edge_theilsen_tform, linewidth=3, color="g")
    plt.plot(quadrant_C.v_edge_theilsen_tform, linewidth=3, color="g")
    plt.plot(quadrant_D.h_edge_theilsen_tform, linewidth=3, color="g")
    plt.plot(quadrant_D.v_edge_theilsen_tform, linewidth=3, color="g")
    plt.show()

    return


def plot_ga_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, best_solution):
    """
    Plotting function to plot the transformation of the quadrants which was found by the
    genetic algorithm.
    """

    # Extract individual tforms from best solution
    ga_tform_A = best_solution[:3]
    ga_tform_B = best_solution[3:6]
    ga_tform_C = best_solution[6:9]
    ga_tform_D = best_solution[9:]

    """
    # Get box corners before GA
    bbox_corners_a = [quadrant_A.bbox_corner_a, quadrant_A.bbox_corner_b,
                      quadrant_A.bbox_corner_c, quadrant_A.bbox_corner_d]
    bbox_corners_b = [quadrant_B.bbox_corner_a, quadrant_B.bbox_corner_b,
                      quadrant_B.bbox_corner_c, quadrant_B.bbox_corner_d]
    bbox_corners_c = [quadrant_C.bbox_corner_a, quadrant_C.bbox_corner_b,
                      quadrant_C.bbox_corner_c, quadrant_C.bbox_corner_d]
    bbox_corners_d = [quadrant_D.bbox_corner_a, quadrant_D.bbox_corner_b,
                      quadrant_D.bbox_corner_c, quadrant_D.bbox_corner_d]

    # Compute center of mass of bbox before GA
    center_a_pre = np.mean(bbox_corners_a, axis=0)
    center_b_pre = np.mean(bbox_corners_b, axis=0)
    center_c_pre = np.mean(bbox_corners_c, axis=0)
    center_d_pre = np.mean(bbox_corners_d, axis=0)

    # Tform the bbox corners with just the rotation.
    rot_tform_a = EuclideanTransform(rotation=-math.radians(ga_tform_A[2]), translation=(0, 0))
    rot_tform_b = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=(0, 0))
    rot_tform_c = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=(0, 0))
    rot_tform_d = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=(0, 0))

    rot_bbox_corners_a = np.squeeze(matrix_transform(bbox_corners_a, rot_tform_a.params))
    rot_bbox_corners_b = np.squeeze(matrix_transform(bbox_corners_b, rot_tform_b.params))
    rot_bbox_corners_c = np.squeeze(matrix_transform(bbox_corners_c, rot_tform_c.params))
    rot_bbox_corners_d = np.squeeze(matrix_transform(bbox_corners_d, rot_tform_d.params))

    # Compute center of mass from tformed bbox corners
    center_a_post = np.mean(rot_bbox_corners_a, axis=0)
    center_b_post = np.mean(rot_bbox_corners_b, axis=0)
    center_c_post = np.mean(rot_bbox_corners_c, axis=0)
    center_d_post = np.mean(rot_bbox_corners_d, axis=0)

    # Account for the offset in translation induced by the rotation
    trans_a = center_a_pre - center_a_post
    trans_b = center_b_pre - center_b_post
    trans_c = center_c_pre - center_c_post
    trans_d = center_d_pre - center_d_post

    # Update the final tform
    final_trans_a = ga_tform_A[:2] + trans_a
    final_trans_b = ga_tform_B[:2] + trans_b
    final_trans_c = ga_tform_C[:2] + trans_c
    final_trans_d = ga_tform_D[:2] + trans_d

    # Apply final tform to the image
    A_tform = EuclideanTransform(rotation=-math.radians(ga_tform_A[2]), translation=final_trans_a)
    B_tform = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=final_trans_b)
    C_tform = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=final_trans_c)
    D_tform = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=final_trans_d)

    """

    A_tform = EuclideanTransform(rotation=-math.radians(ga_tform_A[2]), translation=ga_tform_A[:2])
    B_tform = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=ga_tform_B[:2])
    C_tform = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=ga_tform_C[:2])
    D_tform = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=ga_tform_D[:2])

    quadrant_A.ga_tform_image = warp(quadrant_A.tform_image, A_tform.inverse)
    quadrant_B.ga_tform_image = warp(quadrant_B.tform_image, B_tform.inverse)
    quadrant_C.ga_tform_image = warp(quadrant_C.tform_image, C_tform.inverse)
    quadrant_D.ga_tform_image = warp(quadrant_D.tform_image, D_tform.inverse)

    # Stitch all images together
    combi_before = quadrant_A.tform_image + quadrant_B.tform_image + quadrant_C.tform_image + quadrant_D.tform_image
    combi_after = quadrant_A.ga_tform_image + quadrant_B.ga_tform_image + quadrant_C.ga_tform_image + quadrant_D.ga_tform_image

    # Show result
    plt.figure()
    plt.subplot(121)
    plt.title("Before GA")
    plt.imshow(combi_before, cmap="gray")

    plt.subplot(122)
    plt.title("After GA")
    plt.imshow(combi_after, cmap="gray")
    plt.show()

    return



