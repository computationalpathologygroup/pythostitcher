import numpy as np
import matplotlib.pyplot as plt
import math
import imageio
import os
import glob
import warnings

from skimage.transform import EuclideanTransform, matrix_transform, warp

from .recombine_quadrants import recombine_quadrants
from .recompute_transform import recompute_transform


def plot_rotation_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D):
    """
    Custom function to plot the result of the automatic rotation
    """

    # Get x/y values of bounding box around quadrant
    A_corners_x = [c[0] for c in quadrant_A.bbox_corners] + [quadrant_A.bbox_corners[0][0]]
    A_corners_y = [c[1] for c in quadrant_A.bbox_corners] + [quadrant_A.bbox_corners[0][1]]
    B_corners_x = [c[0] for c in quadrant_B.bbox_corners] + [quadrant_B.bbox_corners[0][0]]
    B_corners_y = [c[1] for c in quadrant_B.bbox_corners] + [quadrant_B.bbox_corners[0][1]]
    C_corners_x = [c[0] for c in quadrant_C.bbox_corners] + [quadrant_C.bbox_corners[0][0]]
    C_corners_y = [c[1] for c in quadrant_C.bbox_corners] + [quadrant_C.bbox_corners[0][1]]
    D_corners_x = [c[0] for c in quadrant_D.bbox_corners] + [quadrant_D.bbox_corners[0][0]]
    D_corners_y = [c[1] for c in quadrant_D.bbox_corners] + [quadrant_D.bbox_corners[0][1]]

    ## Plot quadrant A
    plt.figure(figsize=(6, 14))
    plt.suptitle(f"Quadrants before and after automatic rotation", fontsize=16)
    plt.subplot(421)
    plt.axis("off")
    plt.title(quadrant_A.quadrant_name)
    plt.imshow(quadrant_A.mask, cmap="gray")
    plt.scatter(quadrant_A.mask_corner_a[0], quadrant_A.mask_corner_a[1], facecolor="r", s=100)
    plt.plot(A_corners_x, A_corners_y, linewidth=4, c="r")
    plt.subplot(422)
    plt.axis("off")
    plt.title(quadrant_A.quadrant_name)
    plt.imshow(np.pad(quadrant_A.rot_mask, [quadrant_A.initial_pad, quadrant_A.initial_pad]), cmap="gray")

    ## Plot quadrant B
    plt.subplot(423)
    plt.axis("off")
    plt.title(quadrant_B.quadrant_name)
    plt.imshow(quadrant_B.mask, cmap="gray")
    plt.scatter(quadrant_B.mask_corner_a[0], quadrant_B.mask_corner_a[1], facecolor="r", s=100)
    plt.plot(B_corners_x, B_corners_y, linewidth=4, c="r")
    plt.subplot(424)
    plt.axis("off")
    plt.title(quadrant_B.quadrant_name)
    plt.imshow(np.pad(quadrant_B.rot_mask, [quadrant_B.initial_pad, quadrant_B.initial_pad]), cmap="gray")

    ## Plot quadrant C
    plt.subplot(425)
    plt.axis("off")
    plt.title(quadrant_C.quadrant_name)
    plt.imshow(quadrant_C.mask, cmap="gray")
    plt.scatter(quadrant_C.mask_corner_a[0], quadrant_C.mask_corner_a[1], facecolor="r", s=100)
    plt.plot(C_corners_x, C_corners_y, linewidth=4, c="r")
    plt.subplot(426)
    plt.axis("off")
    plt.title(quadrant_C.quadrant_name)
    plt.imshow(np.pad(quadrant_C.rot_mask, [quadrant_C.initial_pad, quadrant_C.initial_pad]), cmap="gray")

    ## Plot quadrant D
    plt.subplot(427)
    plt.axis("off")
    plt.title(quadrant_D.quadrant_name)
    plt.imshow(quadrant_D.mask, cmap="gray")
    plt.scatter(quadrant_D.mask_corner_a[0], quadrant_D.mask_corner_a[1], facecolor="r", s=100)
    plt.plot(D_corners_x, D_corners_y, linewidth=4, c="r")
    plt.subplot(428)
    plt.axis("off")
    plt.title(quadrant_D.quadrant_name)
    plt.imshow(np.pad(quadrant_D.rot_mask, [quadrant_D.initial_pad, quadrant_D.initial_pad]), cmap="gray")
    plt.tight_layout()
    plt.show()

    return


def plot_transformation_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters):
    """
    Custom function to plot the result of the initial transformation to align the pieces
    """

    result = recombine_quadrants(quadrant_A.colour_image_temp, quadrant_B.colour_image_temp,
                                 quadrant_C.colour_image_temp, quadrant_D.colour_image_temp)

    imsavedir = f"{parameters['results_dir']}/{parameters['patient_idx']}/images/initial_result.png"

    fig = plt.figure()
    current_res = parameters['resolutions'][parameters['iteration']]
    plt.title(f"Initial alignment at resolution {current_res}")
    plt.imshow(result, cmap="gray")
    if parameters["iteration"] == 0:
        plt.savefig(imsavedir)
    plt.close(fig)


def plot_theilsen_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters):
    """
    Custom function to plot the result of the Theil-Sen line estimator
    """

    # Combine all images
    combi_image = recombine_quadrants(quadrant_A.colour_image_temp, quadrant_B.colour_image_temp,
                                      quadrant_C.colour_image_temp, quadrant_D.colour_image_temp)
    ratio =  parameters["resolutions"][parameters["iteration"]]/parameters["resolutions"][0]
    ms = np.sqrt(2500*np.sqrt(ratio))

    # Plot theilsen lines with marked endpoints
    plt.figure()
    plt.title(f"Alignment at resolution {parameters['resolutions'][parameters['iteration']]} \n before genetic algorithm")
    plt.imshow(combi_image, cmap="gray")
    plt.plot(quadrant_A.v_edge_theilsen_endpoints[:, 0], quadrant_A.v_edge_theilsen_endpoints[:, 1],
             linewidth=2, color="g")
    plt.plot(quadrant_A.h_edge_theilsen_endpoints[:, 0], quadrant_A.h_edge_theilsen_endpoints[:, 1],
             linewidth=2, color="b")
    plt.plot(quadrant_B.v_edge_theilsen_endpoints[:, 0], quadrant_B.v_edge_theilsen_endpoints[:, 1],
             linewidth=2, color="g")
    plt.plot(quadrant_B.h_edge_theilsen_endpoints[:, 0], quadrant_B.h_edge_theilsen_endpoints[:, 1],
             linewidth=2, color="b")
    plt.plot(quadrant_C.v_edge_theilsen_endpoints[:, 0], quadrant_C.v_edge_theilsen_endpoints[:, 1],
             linewidth=2, color="g")
    plt.plot(quadrant_C.h_edge_theilsen_endpoints[:, 0], quadrant_C.h_edge_theilsen_endpoints[:, 1],
             linewidth=2, color="b")
    plt.plot(quadrant_D.v_edge_theilsen_endpoints[:, 0], quadrant_D.v_edge_theilsen_endpoints[:, 1],
             linewidth=2, color="g")
    plt.plot(quadrant_D.h_edge_theilsen_endpoints[:, 0], quadrant_D.h_edge_theilsen_endpoints[:, 1],
             linewidth=2, color="b")

    plt.scatter(quadrant_A.v_edge_theilsen_endpoints[:, 0], quadrant_A.v_edge_theilsen_endpoints[:, 1],
                marker='*', s=ms, color="g")
    plt.scatter(quadrant_A.h_edge_theilsen_endpoints[:, 0], quadrant_A.h_edge_theilsen_endpoints[:, 1],
                marker='+', s=ms, color="b")
    plt.scatter(quadrant_B.v_edge_theilsen_endpoints[:, 0], quadrant_B.v_edge_theilsen_endpoints[:, 1],
                marker='*', s=ms, color="g")
    plt.scatter(quadrant_B.h_edge_theilsen_endpoints[:, 0], quadrant_B.h_edge_theilsen_endpoints[:, 1],
                marker='+', s=ms, color="b")
    plt.scatter(quadrant_C.v_edge_theilsen_endpoints[:, 0], quadrant_C.v_edge_theilsen_endpoints[:, 1],
                marker='*', s=ms, color="g")
    plt.scatter(quadrant_C.h_edge_theilsen_endpoints[:, 0], quadrant_C.h_edge_theilsen_endpoints[:, 1],
                marker='+', s=ms, color="b")
    plt.scatter(quadrant_D.v_edge_theilsen_endpoints[:, 0], quadrant_D.v_edge_theilsen_endpoints[:, 1],
                marker='*', s=ms, color="g")
    plt.scatter(quadrant_D.h_edge_theilsen_endpoints[:, 0], quadrant_D.h_edge_theilsen_endpoints[:, 1],
                marker='+', s=ms, color="b")

    plt.legend(["v edge", "h edge"])
    plt.show()

    return


def plot_rotated_bbox(quadrant_A, quadrant_B, quadrant_C, quadrant_D):
    """
    Custom function to plot the bounding box points after the box has been rotated to become horizontal
    """

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


def plot_ga_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, final_tform, ga_dict):
    """
    Plotting function to plot the transformation of the quadrants which was found by the
    genetic algorithm.
    """

    # Extract individual tforms from final_tform
    ga_tform_A = final_tform[quadrant_A.quadrant_name]
    ga_tform_B = final_tform[quadrant_B.quadrant_name]
    ga_tform_C = final_tform[quadrant_C.quadrant_name]
    ga_tform_D = final_tform[quadrant_D.quadrant_name]

    # Make transform object
    A_tform = EuclideanTransform(rotation=-math.radians(ga_tform_A[2]), translation=ga_tform_A[:2])
    B_tform = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=ga_tform_B[:2])
    C_tform = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=ga_tform_C[:2])
    D_tform = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=ga_tform_D[:2])

    # Apply transformation
    quadrant_A.ga_tform_image = warp(quadrant_A.colour_image, A_tform.inverse, output_shape=ga_tform_A[3])
    quadrant_B.ga_tform_image = warp(quadrant_B.colour_image, B_tform.inverse, output_shape=ga_tform_B[3])
    quadrant_C.ga_tform_image = warp(quadrant_C.colour_image, C_tform.inverse, output_shape=ga_tform_C[3])
    quadrant_D.ga_tform_image = warp(quadrant_D.colour_image, D_tform.inverse, output_shape=ga_tform_D[3])

    # Stitch all images together
    result = recombine_quadrants(quadrant_A.ga_tform_image, quadrant_B.ga_tform_image,
                                 quadrant_C.ga_tform_image, quadrant_D.ga_tform_image)

    res = parameters["resolutions"][parameters["iteration"]]
    imsavedir = f"{parameters['results_dir']}/{parameters['patient_idx']}/images/ga_result_{res}.png"

    # Show result
    plt.figure()
    plt.title(f"Alignment at resolution {res}\n after genetic algorithm (fitness={np.round(ga_dict['solution_fitness'], 3)})")
    plt.imshow(result, cmap="gray")
    plt.savefig(imsavedir)
    plt.show()

    return


def make_tform_gif(parameters):
    """
    Custom function to make a gif of all the tformed results after the GA as a visual check of the result
    """


    """
    for sol, gen, fit in zip(solutions, generations, fitness):
        tform_A, tform_B, tform_C, tform_D = recompute_transform(quadrant_A, quadrant_B, quadrant_C, quadrant_D, sol)

        # Make transform object
        tform_A_object = EuclideanTransform(rotation=-math.radians(tform_A[2]), translation=tform_A[:2])
        tform_B_object = EuclideanTransform(rotation=-math.radians(tform_B[2]), translation=tform_B[:2])
        tform_C_object = EuclideanTransform(rotation=-math.radians(tform_C[2]), translation=tform_C[:2])
        tform_D_object = EuclideanTransform(rotation=-math.radians(tform_D[2]), translation=tform_D[:2])

        # Apply transformation
        ga_tform_image_A = warp(quadrant_A.tform_image, tform_A_object.inverse)
        ga_tform_image_B = warp(quadrant_B.tform_image, tform_B_object.inverse)
        ga_tform_image_C = warp(quadrant_C.tform_image, tform_C_object.inverse)
        ga_tform_image_D = warp(quadrant_D.tform_image, tform_D_object.inverse)

        result = recombine_quadrants(ga_tform_image_A, ga_tform_image_B, ga_tform_image_C, ga_tform_image_D)

        plt.figure()
        plt.title(f"Generation {gen}: fitness {fit}")
        plt.imshow(result, cmap="gray")
        plt.axis("off")
        if make_gif:
            plt.savefig(f"{imsavedir}/gen_{str(gen).zfill(4)}.png")
        plt.show()
    """

    # Make gif of the transformation
    imsavedir = f"{parameters['results_dir']}/{parameters['patient_idx']}/images"
    gifsavedir = f"{parameters['results_dir']}/{parameters['patient_idx']}/tform_progression.gif"

    all_images = glob.glob(imsavedir+"/*")
    all_images = sorted(all_images, key=lambda t: os.stat(t).st_mtime)

    images = []
    for name in all_images:
        image = imageio.imread(name)
        images.append(image)

    imageio.mimsave(gifsavedir, images, duration=0.5)

    return


def plot_ga_multires(parameters):
    """
    Custom function to plot how the fitness improves at multiple resolutions.

    NOTE: The fitness depends on the cost function being used and may not scale correctly with the resolutions. This
    may result in a decreasing fitness for higher resolutions while the visual fitness increases. Example: absolute
    distance between the endpoints of the edges increases for higher resolutions leading to a lower fitness when
    this is the only factor in the cost function.
    """

    fitness = parameters["GA_fitness"]
    xticks_loc = list(np.arange(0, 5))
    xticks_label = ["Initial"] + parameters["resolutions"]

    # Only plot when the GA fitness has been tracked properly.
    if len(fitness) == len(xticks_label):
        plt.figure()
        plt.title("Fitness progression at multiple resolutions")
        plt.plot(xticks_loc, fitness)
        plt.xlabel("Resolution")
        plt.xticks(xticks_loc, xticks_label)
        plt.ylabel("Fitness")
        plt.show()
    else:
        warnings.warn("Could not plot fitness progression for multiple resolutions, "
                      "try running the genetic algorithm from scratch by deleting previously acquired tform files")

    return

