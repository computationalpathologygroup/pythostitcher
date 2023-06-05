import glob
import os
import warnings
import numpy as np
import imageio
import matplotlib.pyplot as plt

from .fuse_images_lowres import fuse_images_lowres
from .get_resname import get_resname


def plot_rotation_result(fragments, parameters):
    """
    Custom function to plot the result of the automatic rotation of all fragments.

    Input:
        - All fragments

    Output:
        - Figure displaying the rotation
    """

    # Get pad values for every image
    fragment_shapes = [f.mask.shape for f in fragments]
    max_shape = np.max(fragment_shapes, axis=0)
    pad_mask = [((max_shape - f.mask.shape) / 2).astype(int) for f in fragments]
    pad_rot_mask = [((max_shape - f.rot_mask.shape) / 2).astype(int) for f in fragments]

    # Apply padding
    padded_mask = [
        np.pad(f.mask, [[p[0], p[0]], [p[1], p[1]]]) for p, f in zip(pad_mask, fragments)
    ]
    padded_rot_mask = [
        np.pad(f.rot_mask, [[p[0], p[0]], [p[1], p[1]]]) for p, f in zip(pad_rot_mask, fragments)
    ]

    # Get x/y values of bounding box around fragment
    corners_x = [
        [c[0] + p[1] for c in f.bbox_corners] + [f.bbox_corners[0][0] + p[1]]
        for p, f in zip(pad_mask, fragments)
    ]
    corners_y = [
        [c[1] + p[0] for c in f.bbox_corners] + [f.bbox_corners[0][1] + p[0]]
        for p, f in zip(pad_mask, fragments)
    ]

    # Plot rotation result
    plt.figure(figsize=(6, len(fragments) * 3))
    plt.suptitle("Fragments before and after \nautomatic rotation", fontsize=20)

    for c, (pad, p_mask, p_rmask, c_x, c_y, f) in enumerate(
        zip(pad_mask, padded_mask, padded_rot_mask, corners_x, corners_y, fragments), 1
    ):
        plt.subplot(parameters["n_fragments"], 2, (c * 2) - 1)
        plt.axis("off")
        plt.title(f.final_orientation, fontsize=16)
        plt.imshow(p_mask, cmap="gray")
        plt.scatter(
            f.mask_corner_a[0] + pad[1], f.mask_corner_a[1] + pad[0], facecolor="r", s=100,
        )
        plt.plot(c_x, c_y, linewidth=4, c="r")
        plt.subplot(parameters["n_fragments"], 2, c * 2)
        plt.axis("off")
        plt.title(f.final_orientation, fontsize=16)
        plt.imshow(p_rmask, cmap="gray")
    plt.savefig(f"{parameters['sol_save_dir']}/images/debug/rotation_result.png")
    plt.close()

    return


def plot_transformation_result(fragments, parameters):
    """
    Custom function to plot the result of the initial transformation to globally
    align the fragments.

    Input:
        - All fragments
        - Dict with parameters

    Output:
        - Figure displaying the aligned fragments
    """

    # Merge all individual fragments images into one final image
    images = [f.colour_image for f in fragments]
    result = fuse_images_lowres(images=images, parameters=parameters)

    current_res = parameters["resolutions"][parameters["iteration"]]

    # Plot figure
    plt.figure()
    plt.title(f"Initial alignment at resolution {current_res}")
    plt.imshow(result, cmap="gray")
    if parameters["iteration"] == 0:
        plt.savefig(f"{parameters['sol_save_dir']}/images/" f"ga_progression/initial_alignment.png")
    plt.close()

    return


def plot_theilsen_result(fragments, parameters):
    """
    Custom function to plot the result of the Theil-Sen line approximation of the
    fragments' edges.

    Input:
        - All fragments
        - Dict with parameters

    Output:
        - Figure displaying the Theil-Sen lines for each fragment
    """

    current_res_name = get_resname(parameters["resolutions"][parameters["iteration"]])

    # Combine all images
    images = [f.colour_image for f in fragments]
    combi_image = fuse_images_lowres(images=images, parameters=parameters)

    # Set some plotting parameters
    ratio = parameters["resolution_scaling"][parameters["iteration"]]
    ms = np.sqrt(2500 * np.sqrt(ratio))

    # Plot theilsen lines with marked endpoints
    plt.figure()
    plt.title(
        f"Alignment at resolution {parameters['resolutions'][parameters['iteration']]}"
        f"\n before genetic algorithm"
    )
    plt.imshow(combi_image, cmap="gray")
    for f in fragments:
        if hasattr(f, "v_edge_theilsen_endpoints"):
            plt.plot(
                f.v_edge_theilsen_endpoints[:, 0],
                f.v_edge_theilsen_endpoints[:, 1],
                linewidth=2,
                color="g",
            )
            plt.scatter(
                f.v_edge_theilsen_coords[:, 0],
                f.v_edge_theilsen_coords[:, 1],
                marker="*",
                s=ms,
                color="g",
                label="_nolegend_",
            )
        if hasattr(f, "h_edge_theilsen_endpoints"):
            plt.plot(
                f.h_edge_theilsen_endpoints[:, 0],
                f.h_edge_theilsen_endpoints[:, 1],
                linewidth=2,
                color="b",
            )
            plt.scatter(
                f.h_edge_theilsen_coords[:, 0],
                f.h_edge_theilsen_coords[:, 1],
                marker="+",
                s=ms,
                color="b",
                label="_nolegend_",
            )
    plt.savefig(
        f"{parameters['sol_save_dir']}/images/debug/theilsen_estimate_{current_res_name}.png"
    )
    plt.close()

    return


def plot_rotated_bbox(fragments, parameters):
    """
    Custom function to plot the bounding box points after the box has been rotated.
    This basically offers a sanity check to verify that the corner points have been
    rotated correctly.

    Input:
        - All fragments
        - Parameter dict

    Output:
        - Figure displaying the rotated bounding box points
    """

    current_res_name = get_resname(parameters["resolutions"][parameters["iteration"]])

    # X and y coordinates of the bounding box
    plt.figure()
    plt.title("Rotated bunding box points")

    for c, f in enumerate(fragments):
        scat_x = [
            f.bbox_corner_a[0],
            f.bbox_corner_b[0],
            f.bbox_corner_c[0],
            f.bbox_corner_d[0],
        ]
        scat_y = [
            f.bbox_corner_a[1],
            f.bbox_corner_b[1],
            f.bbox_corner_c[1],
            f.bbox_corner_d[1],
        ]
        plt.subplot(parameters["n_fragments"], 2, c + 1)
        plt.imshow(f.tform_image, cmap="gray")
        plt.scatter(scat_x, scat_y, s=25, c="r")
    plt.savefig(f"{parameters['sol_save_dir']}/images/debug/rotation_bbox_{current_res_name}.png")
    plt.close()

    return


def plot_tformed_edges(fragments, parameters):
    """
    Custom function to plot the transformed edges before inputting them into the
    genetic algorithm. This mainly serves as a sanity check while debugging.

    Input:
        - All fragments

    Output:
        - Figure displaying the transformed edges
    """

    current_res_name = get_resname(parameters["resolutions"][parameters["iteration"]])

    plt.figure()
    plt.title("Transformed edges")
    for f in fragments:
        if hasattr(f, "h_edge_tform"):
            plt.plot(f.h_edge[:, 0], f.h_edge[:, 1], c="b")
        if hasattr(f, "v_edge_tform"):
            plt.plot(f.v_edge[:, 0], f.v_edge[:, 1], c="g")
    plt.legend(["Hor", "Ver"])
    plt.savefig(
        f"{parameters['sol_save_dir']}/images/debug/tformed_edges_inputGA_{current_res_name}.png"
    )
    plt.close()

    return


def plot_tformed_theilsen_lines(fragments, parameters):
    """
    Custom function to plot the transformed Theilsen lines before inputting them
    into the genetic algorithm. This function is analogous to the plot_tformed_edges
    function and serves as a sanity check during debugging.

    Input:
        - All fragments

    Output:
        - Figure displaying the transformed Theil-Sen lines
    """

    current_res_name = get_resname(parameters["resolutions"][parameters["iteration"]])

    plt.figure()
    plt.title("Transformed Theil-Sen lines")
    for f in fragments:
        if hasattr(f, "h_edge_theilsen"):
            plt.plot(f.h_edge_theilsen_tform[:, 0], f.h_edge_theilsen_tform[:, 1], c="b")
        if hasattr(f, "v_edge_theilsen"):
            plt.plot(f.v_edge_theilsen_tform[:, 0], f.v_edge_theilsen_tform[:, 1], c="g")
    plt.legend(["Hor", "Ver"])
    plt.savefig(
        f"{parameters['sol_save_dir']}/images/debug/theilsenlines_inputGA_{current_res_name}.png"
    )
    plt.close()

    return


def plot_ga_tform(fragments, parameters):
    """
    Custom function to show the transformation of the Theil-Sen lines which was found
    by the genetic algorithm.

    Input:
        - All fragments

    Output:
        - Figure displaying the transformed Theil-Sen lines by only taking into account
          the optimal transformation found by the genetic algorithm.
    """

    current_res_name = get_resname(parameters["resolutions"][parameters["iteration"]])

    plt.figure()
    plt.subplot(121)
    plt.title("Theil-Sen lines before GA tform")
    for f in fragments:
        plt.plot(f.h_edge_theilsen_coords, linewidth=3, color="b")
        plt.plot(f.v_edge_theilsen_coords, linewidth=3, color="g")

    plt.subplot(122)
    plt.title("Theil-Sen lines after GA tform")
    for f in fragments:
        plt.plot(f.h_edge_theilsen_tform, linewidth=3, color="b")
        plt.plot(f.v_edge_theilsen_tform, linewidth=3, color="g")
    plt.savefig(
        f"{parameters['sol_save_dir']}/images/debug/theilsenlines_outputGA_{current_res_name}.png"
    )
    plt.close()

    return


def plot_ga_result(final_image, parameters):
    """
    Plotting function to plot the transformation of the fragments which was found
    by the genetic algorithm.

    Input:
        - All fragments
        - Dict with parameters
        - Final transformation from genetic algorithm

    Output:
        - Figure displaying the end result obtained by the genetic algorithm
    """

    current_res_name = get_resname(parameters["resolutions"][parameters["iteration"]])

    # Save result
    plt.figure()
    plt.title(
        f"Alignment at resolution {current_res_name}\n after genetic algorithm "
        f"(fitness={np.round(parameters['GA_fitness'][-1], 2)})"
    )
    plt.imshow(final_image)
    plt.savefig(
        f"{parameters['sol_save_dir']}/images/ga_progression/" f"ga_result_{current_res_name}.png"
    )
    plt.close()

    return


def make_tform_gif(parameters):
    """
    Custom function to make a gif of all the tformed results after the genetic algorithm
    as a visual check of the result. This function requires that intermediate results of
    the genetic algorithm from the different resolutions are saved in a directory called
    images. This function will then combine these images into the GIF.

    Input:
        - Dict with parameters

    Output:
        - GIF file of the transformation
    """

    # Make gif of the transformation
    imsavedir = f"{parameters['sol_save_dir']}/images/ga_progression/"
    gifsavedir = f"{parameters['sol_save_dir']}/images/tform_progression.gif"

    all_images = glob.glob(imsavedir + "/*")
    all_images = sorted(all_images, key=lambda t: os.stat(t).st_mtime)

    images = []
    for name in all_images:
        image = imageio.imread(name)
        images.append(image)

    imageio.mimsave(gifsavedir, images, duration=0.75)

    return


def plot_sampled_patches(total_im, patch_indices_x, patch_indices_y, ts_lines):
    """
    Custom function to visualize the patches which are extracted in the histogram cost
    function. This function serves as a visual check that the patches are extracted
    correctly.

    Input:
        - Final recombined image
        - X/Y indices of the extracted patches
        - Theil-Sen lines

    Output:
        - Figure displaying the sampled patches
    """

    # Plotting parameters
    ts_line_colours = ["b", "g"] * 4

    plt.figure()
    plt.title("Sampled patches on TheilSen lines")
    plt.imshow(total_im, cmap="gray")
    for x, y in zip(patch_indices_x.values(), patch_indices_y.values()):
        plt.plot(x, y, linewidth=0.5, c="r")
    for ts, c in zip(ts_lines, ts_line_colours):
        plt.plot(ts[:, 0], ts[:, 1], linewidth=2, c=c)
    plt.close()

    return


def plot_overlap_cost(im, relative_overlap):
    """
    Custom function to plot the overlap between the fragments. Currently the overlap
    between the fragments is visualized as a rather gray area, this could of course be
    visualized more explicitly.

    Input:
        - Final stitched image
        - Percentual overlap

    Output:
        - Figure displaying the overlap between the fragments
    """

    plt.figure()
    plt.title(f"Visualization of overlapping fragments {np.round(relative_overlap*100, 1)}%")
    plt.imshow(im, cmap="gray")
    plt.close()

    return


def plot_ga_multires(parameters):
    """
    Custom function to plot how the fitness improves at multiple resolutions.

    NOTE: The fitness depends on the cost function being used and may not scale correctly
    with the resolutions. This may result in a decreasing fitness for higher resolutions
    while the visual fitness increases. Example: absolute distance between the endpoints
    of the edges increases for higher resolutions leading to a lower fitness when this is
    the only factor in the cost function.

    Input:
        - Dict with parameters

    Output:
        - Figure displaying the evolution of the fitness over different resolutions.
    """

    # Set some plotting parameters
    fitness = parameters["GA_fitness"]
    xticks_loc = list(np.arange(0, 5))
    xticks_label = ["Initial"] + parameters["resolutions"]

    # Save csv of cost (inverse of fitness, lower cost is better) per res
    df_savepath = parameters["sol_save_dir"].joinpath("tform", "cost_per_res.csv")
    df = pd.DataFrame()
    df["resolution"] = xticks_label
    df["cost"] = [np.round(1/i, 3) for i in fitness]
    df.to_csv(df_savepath, index=False)

    # Only plot when the GA fitness has been tracked properly (e.g. when the cost function
    # has been scaled properly throughout the different resolutions).
    if len(fitness) == len(xticks_label):
        plt.figure()
        plt.title("Fitness progression at multiple resolutions")
        plt.plot(xticks_loc, fitness)
        plt.xlabel("Resolution")
        plt.xticks(xticks_loc, xticks_label)
        plt.ylabel("Fitness")
        plt.savefig(f"{parameters['sol_save_dir']}/images/GA_fitness_result.png")
        plt.close()
    else:
        warnings.warn(
            "Could not plot fitness progression for multiple resolutions, "
            "try running the genetic algorithm from scratch by deleting "
            "the results directory of this patient"
        )

    return
