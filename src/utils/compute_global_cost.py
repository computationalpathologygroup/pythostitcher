import numpy as np

from .rotate_cp import rotate_cp, rotate_2dpoints
from .get_cost_functions import get_cost_functions
from .plot_tools import plot_tformed_edges, plot_tformed_theilsen_lines


def compute_global_cost(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, initial_tform, ga_tform, test, plot):
    """
    Custom function to compute the global cost related to the genetic algorithm. This function takes in
    all quadrants and the general parameters dict and will optimize the adjustment_tform.
    """

    # General cost function parameters
    parameters["intensity_cost_weights"] = 1 - parameters["overunderlap_weights"][quadrant_A.iteration]
    parameters["center_of_mass"] = np.mean(parameters["image_centers"])

    tform_A = initial_tform[quadrant_A.quadrant_name]
    tform_B = initial_tform[quadrant_B.quadrant_name]
    tform_C = initial_tform[quadrant_C.quadrant_name]
    tform_D = initial_tform[quadrant_D.quadrant_name]

    if any([t == [0]*len(tform_A) for t in [tform_A, tform_B, tform_C, tform_D]]):
        raise ValueError("Transformation matrix is empty, please check code")

    ga_tform_B = ga_tform[:3]
    ga_tform_C = ga_tform[3:6]
    ga_tform_D = ga_tform[6:]

    ### Rotate all edges
    # Horizontal edge quadrant B
    edge1B_xy_tformed = rotate_2dpoints(quadrant_B.h_edge, origin=quadrant_B.image_center,
                                        degrees=ga_tform_B[2])
    edge1B_xy_tformed = edge1B_xy_tformed + np.tile([ga_tform_B[0], ga_tform_B[1]],
                                                    [np.shape(edge1B_xy_tformed)[0], 1])
    # quadrant_B.h_edge_tform = edge1B_xy_tformed[:, [1, 0]]
    quadrant_B.h_edge_tform = edge1B_xy_tformed

    # Vertical edge quadrant B
    edge2B_xy_tformed = rotate_2dpoints(quadrant_B.v_edge, origin=quadrant_B.image_center,
                                        degrees=ga_tform_B[2])
    edge2B_xy_tformed = edge2B_xy_tformed + np.tile([ga_tform_B[0], ga_tform_B[1]],
                                                    [np.shape(edge2B_xy_tformed)[0], 1])
    # quadrant_B.v_edge_tform = edge2B_xy_tformed[:, [1, 0]]
    quadrant_B.v_edge_tform = edge2B_xy_tformed

    # Horizontal edge quadrant C
    edge1C_xy_tformed = rotate_2dpoints(quadrant_C.h_edge, origin=quadrant_C.image_center,
                                        degrees=ga_tform_C[2])
    edge1C_xy_tformed = edge1C_xy_tformed + np.tile([ga_tform_C[0], ga_tform_C[1]],
                                                    [np.shape(edge1C_xy_tformed)[0], 1])
    # quadrant_C.h_edge_tform = edge1C_xy_tformed[:, [1, 0]]
    quadrant_C.h_edge_tform = edge1C_xy_tformed

    # Vertical edge quadrant C
    edge2C_xy_tformed = rotate_2dpoints(quadrant_C.v_edge, origin=quadrant_C.image_center,
                                        degrees=ga_tform_C[2])
    edge2C_xy_tformed = edge2C_xy_tformed + np.tile([ga_tform_C[0], ga_tform_C[1]],
                                                    [np.shape(edge2C_xy_tformed)[0], 1])
    # quadrant_C.v_edge_tform = edge2C_xy_tformed[:, [1, 0]]
    quadrant_C.v_edge_tform = edge2C_xy_tformed

    # Horizontal edge quadrant D
    edge1D_xy_tformed = rotate_2dpoints(quadrant_D.h_edge, origin=quadrant_D.image_center,
                                        degrees=ga_tform_D[2])
    edge1D_xy_tformed = edge1D_xy_tformed + np.tile([ga_tform_D[0], ga_tform_D[1]],
                                                    [np.shape(edge1D_xy_tformed)[0], 1])
    # quadrant_D.h_edge_tform = edge1D_xy_tformed[:, [1, 0]]
    quadrant_D.h_edge_tform = edge1D_xy_tformed

    # Vertical edge quadrant D
    edge2D_xy_tformed = rotate_2dpoints(quadrant_D.v_edge, origin=quadrant_D.image_center,
                                        degrees=ga_tform_D[2])
    edge2D_xy_tformed = edge2D_xy_tformed + np.tile([ga_tform_D[0], ga_tform_D[1]],
                                                    [np.shape(edge2D_xy_tformed)[0], 1])
    # quadrant_D.v_edge_tform = edge2D_xy_tformed[:, [1, 0]]
    quadrant_D.v_edge_tform = edge2D_xy_tformed

    ### Rotate lines
    # Horizontal TS line quadrant B
    line1B_xy_tformed = rotate_2dpoints(quadrant_B.h_edge_theilsen_coords, origin=quadrant_B.image_center,
                                        degrees=ga_tform_B[2])
    quadrant_B.h_edge_theilsen_tform = line1B_xy_tformed + np.tile([ga_tform_B[0], ga_tform_B[1]],
                                                                   [np.shape(line1B_xy_tformed)[0], 1])

    # Vertical TS line quadrant B
    line2B_xy_tformed = rotate_2dpoints(quadrant_B.v_edge_theilsen_coords, origin=quadrant_B.image_center,
                                        degrees=ga_tform_B[2])
    quadrant_B.v_edge_theilsen_tform = line2B_xy_tformed + np.tile([ga_tform_B[0], ga_tform_B[1]],
                                                                   [np.shape(line2B_xy_tformed)[0], 1])

    # Horizontal TS line quadrant C
    line1C_xy_tformed = rotate_2dpoints(quadrant_C.h_edge_theilsen_coords, origin=quadrant_C.image_center,
                                        degrees=ga_tform_C[2])
    quadrant_C.h_edge_theilsen_tform = line1C_xy_tformed + np.tile([ga_tform_C[0], ga_tform_C[1]],
                                                                   [np.shape(line1C_xy_tformed)[0], 1])

    # Vertical TS line quadrant C
    line2C_xy_tformed = rotate_2dpoints(quadrant_C.v_edge_theilsen_coords, origin=quadrant_C.image_center,
                                        degrees=ga_tform_C[2])
    quadrant_C.v_edge_theilsen_tform = line2C_xy_tformed + np.tile([ga_tform_C[0], ga_tform_C[1]],
                                                                   [np.shape(line2C_xy_tformed)[0], 1])

    # Horizontal TS line quadrant D
    line1D_xy_tformed = rotate_2dpoints(quadrant_D.h_edge_theilsen_coords, origin=quadrant_D.image_center,
                                        degrees=ga_tform_D[2])
    quadrant_D.h_edge_theilsen_tform = line1D_xy_tformed + np.tile([ga_tform_D[0], ga_tform_D[1]],
                                                                    [np.shape(line1D_xy_tformed)[0], 1])

    # Vertical TS line quadrant D
    line2D_xy_tformed = rotate_2dpoints(quadrant_D.v_edge_theilsen_coords, origin=quadrant_D.image_center,
                                        degrees=ga_tform_D[2])
    quadrant_D.v_edge_theilsen_tform = line2D_xy_tformed + np.tile([ga_tform_D[0], ga_tform_D[1]],
                                                                   [np.shape(line2D_xy_tformed)[0], 1])

    ### Continue here ###

    # Packup some variables to loop over for computing global cost
    whichQuadrants = ['topBottom', 'topBottom', 'bottomTop', 'bottomTop',
                      'leftRight', 'rightLeft', 'leftRight', 'rightLeft']

    edges_tform = [[quadrant_A.h_edge, quadrant_C.h_edge],
                   [quadrant_B.h_edge, quadrant_D.h_edge],
                   [quadrant_C.h_edge, quadrant_A.h_edge],
                   [quadrant_D.h_edge, quadrant_B.h_edge],
                   [quadrant_A.v_edge, quadrant_B.v_edge],
                   [quadrant_B.v_edge, quadrant_A.v_edge],
                   [quadrant_C.v_edge, quadrant_D.v_edge],
                   [quadrant_D.v_edge, quadrant_C.v_edge]]

    edges_theilsen_tform = [[quadrant_A.h_edge_theilsen_coords,  quadrant_C.h_edge_theilsen_coords],
                        [quadrant_B.h_edge_theilsen_coords,  quadrant_D.h_edge_theilsen_coords],
                        [quadrant_C.h_edge_theilsen_coords, quadrant_A.h_edge_theilsen_coords],
                        [quadrant_D.h_edge_theilsen_coords, quadrant_B.h_edge_theilsen_coords],
                        [quadrant_A.v_edge_theilsen_coords, quadrant_B.v_edge_theilsen_coords],
                        [quadrant_B.v_edge_theilsen_coords, quadrant_A.v_edge_theilsen_coords],
                        [quadrant_C.v_edge_theilsen_coords, quadrant_D.v_edge_theilsen_coords],
                        [quadrant_D.v_edge_theilsen_coords, quadrant_C.v_edge_theilsen_coords]]

    intensities = [[quadrant_A.intensities_h, quadrant_C.intensities_h],
                  [quadrant_B.intensities_h, quadrant_D.intensities_h],
                  [quadrant_C.intensities_h, quadrant_A.intensities_h],
                  [quadrant_D.intensities_h, quadrant_B.intensities_h],
                  [quadrant_A.intensities_v, quadrant_B.intensities_v],
                  [quadrant_B.intensities_v, quadrant_A.intensities_v],
                  [quadrant_C.intensities_v, quadrant_D.intensities_v],
                  [quadrant_D.intensities_v, quadrant_C.intensities_v]]

    histograms = [[quadrant_A.hists_h, quadrant_C.hists_h],
                  [quadrant_B.hists_h, quadrant_D.hists_h],
                  [quadrant_C.hists_h, quadrant_A.hists_h],
                  [quadrant_D.hists_h, quadrant_B.hists_h],
                  [quadrant_A.hists_v, quadrant_B.hists_v],
                  [quadrant_B.hists_v, quadrant_A.hists_v],
                  [quadrant_C.hists_v, quadrant_D.hists_v],
                  [quadrant_D.hists_v, quadrant_C.hists_v]]

    # Preallocate dict for saving results
    cost_result_dict = dict()
    cost_result_dict["intensity_costs"] = []
    cost_result_dict["edge_sample_idx"] = []
    cost_result_dict["overunderlap_costs"] = []
    cost_result_dict["overhang_idxs"] = []

    dict_intensity_costs = {}

    # Histogram-based cost function
    if parameters["cost_functions"][quadrant_A.iteration] == "simple_hists":

        for i in range(8):
            intensity_costs, edge_sample_idx, overunderlap_costs, overhang_idxs = get_cost_functions(
                edges_tform[i][0], edges_tform[i][1], edges_theilsen_tform[i][0], edges_theilsen_tform[i][1],
                histograms[i][0], histograms[i][1], whichQuadrants[i], parameters)

            cost_result_dict["intensity_costs"].append(intensity_costs)
            cost_result_dict["overunderlap_costs"].append(overunderlap_costs)
            cost_result_dict["overhang_idxs"].append(overhang_idxs)
            cost_result_dict["edge_sample_idx"].append(edge_sample_idx)

            # Not sure why this variable is needed
            # intensityCosts[i][overhangIdxs[i]] = overhangPenalty

    # Intensity-based cost function
    elif parameters["cost_functions"][quadrant_A.iteration] == "raw_intensities":

        for i in range(8):
            intensity_costs, edge_sample_idx, overunderlap_costs, overhang_idxs = get_cost_functions(
                edges_tform[i][0], edges_tform[i][1], edges_theilsen_tform[i][0], edges_theilsen_tform[i][1],
                intensities[i][0], intensities[i][1], whichQuadrants[i], parameters)

            cost_result_dict["intensity_costs"].append(intensity_costs)
            cost_result_dict["overunderlap_costs"].append(overunderlap_costs)
            cost_result_dict["overhang_idxs"].append(overhang_idxs)
            cost_result_dict["edge_sample_idx"].append(edge_sample_idx)

            # Not sure why this variable is needed
            # intensityCosts[i][overhangIdxs[i]] = overhangPenalty

    else:
        raise ValueError(f"Unexpected cost function, options are: [simpleHists, rawIntensities]")

    # Compute mean and overall costs

    """
    mean_overunderlap_costs = np.zeros((8))
    mean_intensity_costs = np.zeros((8))

    for j in range(8):
        mean_overunderlap_costs[j] = np.mean([value for value in overlapAndUnderlapCosts.values()])
        mean_intensity_costs[j] = np.mean([value for value in intensityCosts.values()])
    
    totalOverlapAndUnderlapCost = np.sqrt(np.mean(mean_overunderlap_costs))
    totalIntensityCost = np.mean(mean_intensity_costs)
    """

    totalOverlapAndUnderlapCost = np.mean(cost_result_dict["overunderlap_costs"])
    totalIntensityCost = np.mean(cost_result_dict["intensity_costs"])

    cost = parameters["intensity_cost_weights"] * totalIntensityCost + \
           parameters["overunderlap_weights"][quadrant_A.iteration] * totalOverlapAndUnderlapCost

    allCosts = dict()
    allCosts["name"] = ["1A", "1B", "1C", "1D", "2A", "2B", "2C", "2D"]
    allCosts["intensityCost"] = totalOverlapAndUnderlapCost
    allCosts["overlapAndUnderlapCost"] = totalIntensityCost

    return cost


def fitness_func_old(solution, solution_idx):
    """
    Custom function to compute the cost related to the genetic algorithm. The genetic algorithm will provide
    a solution for the transformation matrix which can be used to compute the cost. A better transformation
    matrix should be associated with a lower cost. Cost should always be in the range [0, 2].
    """

    # General cost function costfun_parameters
    costfun_parameters["intensity_cost_weights"] = 1 - costfun_parameters["overunderlap_weights"][costfun_quadrant_A.iteration]
    costfun_parameters["center_of_mass"] = np.mean(costfun_parameters["image_centers"])

    # Get transformation per quadrant from ga solution
    ga_tform_B = solution[:3]
    ga_tform_C = solution[3:6]
    ga_tform_D = solution[6:]

    ### Rotate all edges
    # Horizontal edge quadrant B
    edge1B_xy_tformed = rotate_2dpoints(costfun_quadrant_B.h_edge, origin=costfun_quadrant_B.image_center,
                                        degrees=ga_tform_B[2])
    edge1B_tform = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=(ga_tform_B[0], ga_tform_B[1]))
    costfun_quadrant_B.h_edge_tform = matrix_transform(costfun_quadrant_B.h_edge, edge1B_tform.params)

    """
    edge1B_xy_tformed = edge1B_xy_tformed + np.tile([ga_tform_B[0], ga_tform_B[1]],
                                                    [np.shape(edge1B_xy_tformed)[0], 1])
    """

    # Vertical edge quadrant B
    edge2B_xy_tformed = rotate_2dpoints(costfun_quadrant_B.v_edge, origin=costfun_quadrant_B.image_center,
                                        degrees=ga_tform_B[2])
    edge2B_xy_tformed = edge2B_xy_tformed + np.tile([ga_tform_B[0], ga_tform_B[1]],
                                                    [np.shape(edge2B_xy_tformed)[0], 1])
    costfun_quadrant_B.v_edge_tform = edge2B_xy_tformed

    # Horizontal edge quadrant C
    edge1C_xy_tformed = rotate_2dpoints(costfun_quadrant_C.h_edge, origin=costfun_quadrant_C.image_center,
                                        degrees=ga_tform_C[2])
    edge1C_xy_tformed = edge1C_xy_tformed + np.tile([ga_tform_C[0], ga_tform_C[1]],
                                                    [np.shape(edge1C_xy_tformed)[0], 1])
    costfun_quadrant_C.h_edge_tform = edge1C_xy_tformed

    # Vertical edge quadrant C
    edge2C_xy_tformed = rotate_2dpoints(costfun_quadrant_C.v_edge, origin=costfun_quadrant_C.image_center,
                                        degrees=ga_tform_C[2])
    edge2C_xy_tformed = edge2C_xy_tformed + np.tile([ga_tform_C[0], ga_tform_C[1]],
                                                    [np.shape(edge2C_xy_tformed)[0], 1])
    costfun_quadrant_C.v_edge_tform = edge2C_xy_tformed

    # Horizontal edge quadrant D
    edge1D_xy_tformed = rotate_2dpoints(costfun_quadrant_D.h_edge, origin=costfun_quadrant_D.image_center,
                                        degrees=ga_tform_D[2])
    edge1D_xy_tformed = edge1D_xy_tformed + np.tile([ga_tform_D[0], ga_tform_D[1]],
                                                    [np.shape(edge1D_xy_tformed)[0], 1])
    costfun_quadrant_D.h_edge_tform = edge1D_xy_tformed

    # Vertical edge quadrant D
    edge2D_xy_tformed = rotate_2dpoints(costfun_quadrant_D.v_edge, origin=costfun_quadrant_D.image_center,
                                        degrees=ga_tform_D[2])
    edge2D_xy_tformed = edge2D_xy_tformed + np.tile([ga_tform_D[0], ga_tform_D[1]],
                                                    [np.shape(edge2D_xy_tformed)[0], 1])
    costfun_quadrant_D.v_edge_tform = edge2D_xy_tformed

    ### Rotate lines
    # Horizontal TS line quadrant B
    line1B_xy_tformed = rotate_2dpoints(costfun_quadrant_B.h_edge_theilsen_coords, origin=costfun_quadrant_B.image_center,
                                        degrees=ga_tform_B[2])
    costfun_quadrant_B.h_edge_theilsen_tform = line1B_xy_tformed + np.tile([ga_tform_B[0], ga_tform_B[1]],
                                                                   [np.shape(line1B_xy_tformed)[0], 1])

    # Vertical TS line quadrant B
    line2B_xy_tformed = rotate_2dpoints(costfun_quadrant_B.v_edge_theilsen_coords, origin=costfun_quadrant_B.image_center,
                                        degrees=ga_tform_B[2])
    costfun_quadrant_B.v_edge_theilsen_tform = line2B_xy_tformed + np.tile([ga_tform_B[0], ga_tform_B[1]],
                                                                   [np.shape(line2B_xy_tformed)[0], 1])

    # Horizontal TS line quadrant C
    line1C_xy_tformed = rotate_2dpoints(costfun_quadrant_C.h_edge_theilsen_coords, origin=costfun_quadrant_C.image_center,
                                        degrees=ga_tform_C[2])
    costfun_quadrant_C.h_edge_theilsen_tform = line1C_xy_tformed + np.tile([ga_tform_C[0], ga_tform_C[1]],
                                                                   [np.shape(line1C_xy_tformed)[0], 1])

    # Vertical TS line quadrant C
    line2C_xy_tformed = rotate_2dpoints(costfun_quadrant_C.v_edge_theilsen_coords, origin=costfun_quadrant_C.image_center,
                                        degrees=ga_tform_C[2])
    costfun_quadrant_C.v_edge_theilsen_tform = line2C_xy_tformed + np.tile([ga_tform_C[0], ga_tform_C[1]],
                                                                   [np.shape(line2C_xy_tformed)[0], 1])

    # Horizontal TS line quadrant D
    line1D_xy_tformed = rotate_2dpoints(costfun_quadrant_D.h_edge_theilsen_coords, origin=costfun_quadrant_D.image_center,
                                        degrees=ga_tform_D[2])
    costfun_quadrant_D.h_edge_theilsen_tform = line1D_xy_tformed + np.tile([ga_tform_D[0], ga_tform_D[1]],
                                                                   [np.shape(line1D_xy_tformed)[0], 1])

    # Vertical TS line quadrant D
    line2D_xy_tformed = rotate_2dpoints(costfun_quadrant_D.v_edge_theilsen_coords, origin=costfun_quadrant_D.image_center,
                                        degrees=ga_tform_D[2])
    costfun_quadrant_D.v_edge_theilsen_tform = line2D_xy_tformed + np.tile([ga_tform_D[0], ga_tform_D[1]],
                                                                   [np.shape(line2D_xy_tformed)[0], 1])

    ### Continue here ###

    # Packup some variables to loop over for computing global cost
    whichQuadrants = ['topBottom', 'topBottom', 'bottomTop', 'bottomTop',
                      'leftRight', 'rightLeft', 'leftRight', 'rightLeft']

    edges_tform = [[costfun_quadrant_A.h_edge, costfun_quadrant_C.h_edge],
                   [costfun_quadrant_B.h_edge, costfun_quadrant_D.h_edge],
                   [costfun_quadrant_C.h_edge, costfun_quadrant_A.h_edge],
                   [costfun_quadrant_D.h_edge, costfun_quadrant_B.h_edge],
                   [costfun_quadrant_A.v_edge, costfun_quadrant_B.v_edge],
                   [costfun_quadrant_B.v_edge, costfun_quadrant_A.v_edge],
                   [costfun_quadrant_C.v_edge, costfun_quadrant_D.v_edge],
                   [costfun_quadrant_D.v_edge, costfun_quadrant_C.v_edge]]

    edges_theilsen_tform = [[costfun_quadrant_A.h_edge_theilsen_coords, costfun_quadrant_C.h_edge_theilsen_coords],
                            [costfun_quadrant_B.h_edge_theilsen_coords, costfun_quadrant_D.h_edge_theilsen_coords],
                            [costfun_quadrant_C.h_edge_theilsen_coords, costfun_quadrant_A.h_edge_theilsen_coords],
                            [costfun_quadrant_D.h_edge_theilsen_coords, costfun_quadrant_B.h_edge_theilsen_coords],
                            [costfun_quadrant_A.v_edge_theilsen_coords, costfun_quadrant_B.v_edge_theilsen_coords],
                            [costfun_quadrant_B.v_edge_theilsen_coords, costfun_quadrant_A.v_edge_theilsen_coords],
                            [costfun_quadrant_C.v_edge_theilsen_coords, costfun_quadrant_D.v_edge_theilsen_coords],
                            [costfun_quadrant_D.v_edge_theilsen_coords, costfun_quadrant_C.v_edge_theilsen_coords]]

    intensities = [[costfun_quadrant_A.intensities_h, costfun_quadrant_C.intensities_h],
                   [costfun_quadrant_B.intensities_h, costfun_quadrant_D.intensities_h],
                   [costfun_quadrant_C.intensities_h, costfun_quadrant_A.intensities_h],
                   [costfun_quadrant_D.intensities_h, costfun_quadrant_B.intensities_h],
                   [costfun_quadrant_A.intensities_v, costfun_quadrant_B.intensities_v],
                   [costfun_quadrant_B.intensities_v, costfun_quadrant_A.intensities_v],
                   [costfun_quadrant_C.intensities_v, costfun_quadrant_D.intensities_v],
                   [costfun_quadrant_D.intensities_v, costfun_quadrant_C.intensities_v]]

    histograms = [[costfun_quadrant_A.hists_h, costfun_quadrant_C.hists_h],
                  [costfun_quadrant_B.hists_h, costfun_quadrant_D.hists_h],
                  [costfun_quadrant_C.hists_h, costfun_quadrant_A.hists_h],
                  [costfun_quadrant_D.hists_h, costfun_quadrant_B.hists_h],
                  [costfun_quadrant_A.hists_v, costfun_quadrant_B.hists_v],
                  [costfun_quadrant_B.hists_v, costfun_quadrant_A.hists_v],
                  [costfun_quadrant_C.hists_v, costfun_quadrant_D.hists_v],
                  [costfun_quadrant_D.hists_v, costfun_quadrant_C.hists_v]]

    # Preallocate dict for saving results
    cost_result_dict = dict()
    cost_result_dict["intensity_costs"] = []
    cost_result_dict["edge_sample_idx"] = []
    cost_result_dict["overunderlap_costs"] = []
    cost_result_dict["overhang_idxs"] = []

    dict_intensity_costs = {}

    # Histogram-based cost function
    if costfun_parameters["cost_functions"][costfun_quadrant_A.iteration] == "simple_hists":

        for i in range(8):
            intensity_costs, edge_sample_idx, overunderlap_costs, overhang_idxs = get_cost_functions(
                edges_tform[i][0], edges_tform[i][1], edges_theilsen_tform[i][0], edges_theilsen_tform[i][1],
                histograms[i][0], histograms[i][1], whichQuadrants[i], costfun_parameters)

            cost_result_dict["intensity_costs"].append(intensity_costs)
            cost_result_dict["overunderlap_costs"].append(overunderlap_costs)
            cost_result_dict["overhang_idxs"].append(overhang_idxs)
            cost_result_dict["edge_sample_idx"].append(edge_sample_idx)

            # Not sure why this variable is needed
            # intensityCosts[i][overhangIdxs[i]] = overhangPenalty

    # Intensity-based cost function
    elif costfun_parameters["cost_functions"][costfun_quadrant_A.iteration] == "raw_intensities":

        for i in range(8):
            intensity_costs, edge_sample_idx, overunderlap_costs, overhang_idxs = get_cost_functions(
                edges_tform[i][0], edges_tform[i][1], edges_theilsen_tform[i][0], edges_theilsen_tform[i][1],
                intensities[i][0], intensities[i][1], whichQuadrants[i], costfun_parameters)

            cost_result_dict["intensity_costs"].append(intensity_costs)
            cost_result_dict["overunderlap_costs"].append(overunderlap_costs)
            cost_result_dict["overhang_idxs"].append(overhang_idxs)
            cost_result_dict["edge_sample_idx"].append(edge_sample_idx)

            # Not sure why this variable is needed
            # intensityCosts[i][overhangIdxs[i]] = overhangPenalty

    else:
        raise ValueError(f"Unexpected cost function, options are: [simpleHists, rawIntensities]")

    # Compute mean and overall costs

    """
    mean_overunderlap_costs = np.zeros((8))
    mean_intensity_costs = np.zeros((8))

    for j in range(8):
        mean_overunderlap_costs[j] = np.mean([value for value in overlapAndUnderlapCosts.values()])
        mean_intensity_costs[j] = np.mean([value for value in intensityCosts.values()])

    totalOverlapAndUnderlapCost = np.sqrt(np.mean(mean_overunderlap_costs))
    totalIntensityCost = np.mean(mean_intensity_costs)
    """

    totalOverlapAndUnderlapCost = np.mean(cost_result_dict["overunderlap_costs"])
    totalIntensityCost = np.mean(cost_result_dict["intensity_costs"])

    cost = costfun_parameters["intensity_cost_weights"] * totalIntensityCost + \
           costfun_parameters["overunderlap_weights"][costfun_quadrant_A.iteration] * totalOverlapAndUnderlapCost

    allCosts = dict()
    allCosts["name"] = ["1A", "1B", "1C", "1D", "2A", "2B", "2C", "2D"]
    allCosts["intensityCost"] = totalOverlapAndUnderlapCost
    allCosts["overlapAndUnderlapCost"] = totalIntensityCost

    return 1/cost
