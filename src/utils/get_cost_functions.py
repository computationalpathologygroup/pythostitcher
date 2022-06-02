import numpy as np
import math
import itertools
import cv2

from shapely.geometry import Polygon
from skimage.transform import EuclideanTransform, matrix_transform, warp
from skimage.measure import label
from skimage.segmentation import find_boundaries

from .plot_tools import plot_sampled_patches, plot_overlap_cost
from .recombine_quadrants import recombine_quadrants


def apply_new_transform(quadrant, tform, tform_image=False):
    """
    Custom function to apply the newly acquired transformation to several attributes of the quadrant.

    Input:
    - Quadrants
    - Transformation matrix

    Output:
    - Transformed quadrants and lines
    """

    # Compute tform object from tform parameters
    tform = EuclideanTransform(rotation=-math.radians(tform[2]), translation=tform[:2])

    # Apply tform to theilsen endpoints
    quadrant.h_edge_theilsen_endpoints_tform = matrix_transform(quadrant.h_edge_theilsen_endpoints, tform.params)
    quadrant.v_edge_theilsen_endpoints_tform = matrix_transform(quadrant.v_edge_theilsen_endpoints, tform.params)

    # Apply tform to mask coordinates
    quadrant.mask_contour_tform = matrix_transform(quadrant.mask_contour, tform.params)

    # Apply tform to image center
    quadrant.image_center = matrix_transform(quadrant.image_center, tform.params)

    # Apply tform to theilsen lines
    #quadrant.h_edge_theilsen_tform = matrix_transform(quadrant.h_edge_theilsen_coords, tform.params)
    #quadrant.v_edge_theilsen_tform = matrix_transform(quadrant.v_edge_theilsen_coords, tform.params)

    # Apply tform to quadrant edges
    #quadrant.h_edge_tform = matrix_transform(quadrant.h_edge, tform.params)
    #quadrant.v_edge_tform = matrix_transform(quadrant.v_edge, tform.params)

    # Apply tform to image when required
    if tform_image:
        quadrant.colour_image = warp(quadrant.colour_image, tform.inverse)

    return quadrant


def hist_cost_function(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, plot=False):
    """
    Custom function to compute the histogram loss component.

    Input:
    - All quadrants
    - Dict with parameters
    - Boolean value whether to plot the result

    Output:
    - Cost between [0, 2]
    """

    # Set starting parameters
    hist_size = parameters["hist_sizes"][parameters["iteration"]]
    step_size = np.round(hist_size/2).astype(int)
    nbins = parameters["nbins"]
    histogram_costs = []
    patch_indices_x = dict()
    patch_indices_y = dict()

    quadrants = [quadrant_A, quadrant_B, quadrant_C, quadrant_D]
    total_im = recombine_quadrants(quadrant_A.tform_image, quadrant_B.tform_image,
                                   quadrant_C.tform_image, quadrant_D.tform_image)

    # Loop over all quadrants to compute the cost for the horizontal and vertical edge
    for quadrant in quadrants:

        # Set the points along horizontal edge to sample
        sample_idx = np.arange(0, len(quadrant.h_edge_theilsen_tform), step=step_size).astype(int)
        sample_locs = quadrant.h_edge_theilsen_tform[sample_idx].astype(int)

        # Create indices of patches on sample points
        patch_idxs_upper = [[x-step_size, x+step_size, y-hist_size, y] for x, y in sample_locs]
        patch_idxs_lower = [[x-step_size, x+step_size, y, y+hist_size] for x, y in sample_locs]

        # Extract patches from the total image
        patches_upper = [total_im[xmin:xmax, ymin:ymax] for xmin, xmax, ymin, ymax in patch_idxs_upper]
        patches_lower = [total_im[xmin:xmax, ymin:ymax] for xmin, xmax, ymin, ymax in patch_idxs_lower]

        # Compute histogram for each patch. By setting the lower range to 0.01 we exclude background pixels.
        histograms_upper = [np.histogram(patch, bins=nbins, range=(0.01, 1)) for patch in patches_upper]
        histograms_lower = [np.histogram(patch, bins=nbins, range=(0.01, 1)) for patch in patches_lower]

        # Compute probability density function for each histogram. Set this to zero if the histogram does not
        # contain any tissue pixels.
        prob_dens_upper = [h[0] / np.sum(h[0]) if np.sum(h[0]) != 0 else np.zeros((nbins)) for h in histograms_upper]
        prob_dens_lower = [h[0] / np.sum(h[0]) if np.sum(h[0]) != 0 else np.zeros((nbins)) for h in histograms_lower]

        # Compute difference between probability density function. For probability density functions of an
        # empty patch we set the cost to 1.
        summed_diff = [1 if (np.sum(prob1) == 0 and np.sum(prob2) == 0) else np.sum(np.abs(prob1 - prob2))
                       for prob1, prob2 in zip(prob_dens_upper, prob_dens_lower)]
        histogram_costs.append(np.mean(summed_diff))

        # Repeat for vertical edge. Set the points along vertical edge to sample
        sample_idx = np.arange(0, len(quadrant.v_edge_theilsen_tform), step=step_size).astype(int)
        sample_locs = quadrant.v_edge_theilsen_tform[sample_idx].astype(int)

        # Create indices of patches on sample points
        patch_idxs_left = [[x-hist_size, x, y-step_size, y+step_size] for x, y in sample_locs]
        patch_idxs_right = [[x, x+hist_size, y-step_size, y+step_size] for x, y in sample_locs]

        # Extract patches from the total image
        patches_left = [total_im[xmin:xmax, ymin:ymax] for xmin, xmax, ymin, ymax in patch_idxs_left]
        patches_right = [total_im[xmin:xmax, ymin:ymax] for xmin, xmax, ymin, ymax in patch_idxs_right]

        # Compute histogram for each patch. By setting the lower range to 0.01 we exclude background pixels.
        histograms_left = [np.histogram(patch, bins=nbins, range=(0.01, 1)) for patch in patches_left]
        histograms_right = [np.histogram(patch, bins=nbins, range=(0.01, 1)) for patch in patches_right]

        # Compute probability density function for each histogram. Set this to zero if the histogram does not
        # contain any tissue pixels.
        prob_dens_left = [h[0] / np.sum(h[0]) if np.sum(h[0]) != 0 else np.zeros((nbins)) for h in histograms_left]
        prob_dens_right = [h[0] / np.sum(h[0]) if np.sum(h[0]) != 0 else np.zeros((nbins)) for h in histograms_right]

        # Compute difference between probability density function. For probability density functions of an
        # empty patch we set the cost to the maximum value of 2.
        summed_diff = [2 if (np.sum(prob1) == 0 and np.sum(prob2) == 0) else np.sum(np.abs(prob1 - prob2))
                       for prob1, prob2 in zip(prob_dens_left, prob_dens_right)]
        histogram_costs.append(np.mean(summed_diff))

        if plot:
            # Save patch indices for plotting
            for patch_idx, s in zip([patch_idxs_upper, patch_idxs_lower, patch_idxs_left, patch_idxs_right],
                                    ["upper", "lower", "left", "right"]):
                xvals = np.array([[x1, x1, x2, x2, x1] for x1, x2, _, _ in patch_idx]).ravel()
                yvals = np.array([[y1, y2, y2, y1, y1] for _, _, y1, y2 in patch_idx]).ravel()
                patch_indices_x[f"{quadrant.quadrant_name}_{s}"] = xvals
                patch_indices_y[f"{quadrant.quadrant_name}_{s}"] = yvals

    # Plot sanity check
    if plot:
        ts_lines = [quadrant_A.h_edge_theilsen_tform,
                    quadrant_A.v_edge_theilsen_tform,
                    quadrant_B.h_edge_theilsen_tform,
                    quadrant_B.v_edge_theilsen_tform,
                    quadrant_C.h_edge_theilsen_tform,
                    quadrant_C.v_edge_theilsen_tform,
                    quadrant_D.h_edge_theilsen_tform,
                    quadrant_D.v_edge_theilsen_tform]
        plot_sampled_patches(total_im, patch_indices_x, patch_indices_y, ts_lines)

    return np.mean(histogram_costs)


def overlap_cost_function(quadrant_A, quadrant_B, quadrant_C, quadrant_D, tforms):
    """
    Custom function to compute the overlap between the quadrants. This is implemented using polygons rather than
    the transformed images as this is an order of magnitude faster.

    Note that the current implementation provides an approximation of the overlap rather than the exact amount as
    overlap is only calculated for quadrant pairs and not quadrant triplets (i.e. if there is overlap between
    quadrant ACD this can be counted multiple times due to inclusion in the AC, AD and CD pairs.
    """

    # Set some initial parameters
    keys = ["A", "B", "C", "D"]
    combinations = itertools.combinations(keys, 2)
    quadrants = [quadrant_A, quadrant_B, quadrant_C, quadrant_D]
    poly_dict = dict()
    total_area = 0
    total_overlap = 0

    # Create a polygon from the transformed mask contour and compute its area
    for quadrant, tform, key in zip(quadrants, tforms, keys):
        poly_dict[key] = Polygon(quadrant.mask_contour_tform)
        total_area += poly_dict[key].area

    # Compute overlap between all possible quadrant pairs
    for combo in combinations:
        overlap_polygon = poly_dict[combo[0]].intersection(poly_dict[combo[1]])
        total_overlap += overlap_polygon.area

    # Compute relative overlap and apply weighting factor
    relative_overlap = total_overlap/total_area
    cost = relative_overlap * 10

    # Verify that the overlap doesn't exceed 100%
    assert relative_overlap < 1, "Overlap cannot be greater than 100%"

    return cost


def distance_cost_function(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters):
    """
    Custom function to compute the distance cost between the endpoints. This distance is computed as the Euclidean
    distance, but we want to scale this to a normalized range in order to account for the increasing distance
    for finer resolutions. We normalize this distance by dividing the distance by the starting value without
    any transformations.

    Input:
    - All quadrants
    - Dict with parameters

    Output:
    - Cost normalized by initial distance
    """

    global distance_scaling

    # Define pairs to loop over
    hline_pairs = [[quadrant_A, quadrant_C], [quadrant_B, quadrant_D]]
    vline_pairs = [[quadrant_A, quadrant_B], [quadrant_C, quadrant_D]]

    distance_costs = []
    scaling = parameters["cost_function_scaling"][parameters["iteration"]]

    for quadrant1, quadrant2 in hline_pairs:

        # Get the lines from the quadrant
        hline1_pts = quadrant1.h_edge_theilsen_endpoints_tform
        hline2_pts = quadrant2.h_edge_theilsen_endpoints_tform

        # Calculate distance from center of mass
        line1_distsFromCoM = [np.linalg.norm(hline1_pts[0] - parameters["center_of_mass"]),
                              np.linalg.norm(hline1_pts[1] - parameters["center_of_mass"])]
        line2_distsFromCoM = [np.linalg.norm(hline2_pts[0] - parameters["center_of_mass"]),
                              np.linalg.norm(hline2_pts[1] - parameters["center_of_mass"])]

        # Get indices of inner and outer points of both lines
        innerPtIdx_line1 = np.argmin(line1_distsFromCoM)
        outerPtIdx_line1 = 1 if innerPtIdx_line1==0 else 0

        innerPtIdx_line2 = np.argmin(line2_distsFromCoM)
        outerPtIdx_line2 = 1 if innerPtIdx_line2==0 else 0

        # Get the inner and outer points. We divide this by the scaling to account for the increased distance due
        # to the higher resolutions.
        line1_innerPt = hline1_pts[innerPtIdx_line1]
        line1_outerPt = hline1_pts[outerPtIdx_line1]
        line2_innerPt = hline2_pts[innerPtIdx_line2]
        line2_outerPt = hline2_pts[outerPtIdx_line2]

        # Compute edge_distance_costs as sum of distances
        inner_point_weight = 1 - parameters["outer_point_weight"]
        inner_point_norm = np.linalg.norm(line1_innerPt - line2_innerPt)**2
        outer_point_norm = np.linalg.norm(line1_outerPt - line2_outerPt)**2
        combined_costs = inner_point_weight * inner_point_norm + parameters["outer_point_weight"] * outer_point_norm
        distance_costs.append(combined_costs)

    for quadrant1, quadrant2 in vline_pairs:

        # Get the lines from the quadrants
        vline1_pts = quadrant1.v_edge_theilsen_endpoints_tform
        vline2_pts = quadrant2.v_edge_theilsen_endpoints_tform

        # Calculate distance from center of mass
        line1_distsFromCoM = [np.linalg.norm(vline1_pts[0] - parameters["center_of_mass"]),
                              np.linalg.norm(vline1_pts[1] - parameters["center_of_mass"])]
        line2_distsFromCoM = [np.linalg.norm(vline2_pts[0] - parameters["center_of_mass"]),
                              np.linalg.norm(vline2_pts[1] - parameters["center_of_mass"])]

        # Get indices of inner and outer points of both lines
        innerPtIdx_line1 = np.argmin(line1_distsFromCoM)
        outerPtIdx_line1 = 1 if innerPtIdx_line1==0 else 0

        innerPtIdx_line2 = np.argmin(line2_distsFromCoM)
        outerPtIdx_line2 = 1 if innerPtIdx_line2==0 else 0

        # Get the inner and outer points. We divide this by the scaling to account for the increased distance due
        # to the higher resolutions.
        line1_innerPt = vline1_pts[innerPtIdx_line1]
        line1_outerPt = vline1_pts[outerPtIdx_line1]
        line2_innerPt = vline2_pts[innerPtIdx_line2]
        line2_outerPt = vline2_pts[outerPtIdx_line2]

        # Compute edge_distance_costs as sum of distances
        inner_point_weight = 1 - parameters["outer_point_weight"]
        inner_point_norm = np.linalg.norm(line1_innerPt - line2_innerPt)**2
        outer_point_norm = np.linalg.norm(line1_outerPt - line2_outerPt)**2
        combined_costs = inner_point_weight * inner_point_norm + parameters["outer_point_weight"] * outer_point_norm
        distance_costs.append(combined_costs)

    if parameters["iteration"] == 0 and parameters["distance_scaling_required"]:
        distance_scaling = np.mean(distance_costs)
        parameters["distance_scaling_required"] = False

    cost = np.mean(distance_costs)/(distance_scaling*scaling)

    return cost
