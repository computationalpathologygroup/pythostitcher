import pygad
import numpy as np
import os
import math
import copy

from skimage.transform import EuclideanTransform, matrix_transform

from .compute_global_cost import compute_global_cost
from .plot_tools import *
from .get_resname import get_resname
from .rotate_cp import rotate_cp, rotate_2dpoints
from .get_cost_functions import get_cost_functions


def genetic_algorithm(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, initial_tform, plot):
    """
    Function that runs a genetic algorithm using the pygad module.
    """

    # Make some variables global for fitness func
    global costfun_quadrant_A, costfun_quadrant_B, costfun_quadrant_C, costfun_quadrant_D
    global costfun_parameters, costfun_initialtform
    costfun_quadrant_A = copy.deepcopy(quadrant_A)
    costfun_quadrant_B = copy.deepcopy(quadrant_B)
    costfun_quadrant_C = copy.deepcopy(quadrant_C)
    costfun_quadrant_D = copy.deepcopy(quadrant_D)
    costfun_parameters = copy.deepcopy(parameters)
    costfun_initialtform = copy.deepcopy(initial_tform)

    # Initialize parameters
    tform_combi = [*initial_tform[quadrant_A.quadrant_name][:-1], *initial_tform[quadrant_B.quadrant_name][:-1],
                   *initial_tform[quadrant_C.quadrant_name][:-1], *initial_tform[quadrant_D.quadrant_name][:-1]]
    # ga_tform = copy.deepcopy(tform_combi)
    num_genes = len(tform_combi)
    ga_tform = np.zeros((num_genes))

    # Cap solution ranges to plausible values
    t_range = parameters["translation_ranges"][parameters["iteration"]]
    a_range = parameters["angle_range"]
    angles = [False, False, True]*4
    lb = [x - a_range if is_angle else x - t_range for x, is_angle in zip(ga_tform, angles)]
    ub = [x + a_range if is_angle else x + t_range for x, is_angle in zip(ga_tform, angles)]
    param_range = [[l, u] for l, u in zip(lb, ub)]

    # Define values for initial population. These should be relatively close to initial tform
    num_sol = 20
    num_gen = 1000
    init_pop = np.zeros((num_sol, num_genes))
    noise_level = parameters["padsizes"][parameters["iteration"]] * 10

    # Generate initial population based on noise
    for i in range(num_sol):
        np.random.seed(i)
        noise = np.random.randint(low=-noise_level, high=noise_level, size=num_genes)/100
        init_pop[i, :] = ga_tform + noise

    # Optimization can consist of many more variables. For now only change these.
    # num_generations, eval_function, num_genes and upper/lower bounds are taken from AutoStitcher.
    # The other parameters are chosen randomly.
    ga_instance = pygad.GA(
        num_generations = num_gen,              # num generations to optimize
        num_parents_mating = 5,                 # num solutions to keep per generation
        fitness_func = fitness_func,            # optimization function
        initial_population = init_pop,          # values for first-gen genes
        gene_space = param_range,               # parameter search range
        mutation_probability=0.3,               # probability that a gene mutates
        mutation_type="random",                 # mutation type
        save_best_solutions=True,
        save_solutions=True
    )

    # Run and show results
    ga_instance.run()
    ga_instance.plot_fitness()

    # Retrieve best results and save
    best_solution, solution_fitness, solution_idx = ga_instance.best_solution()

    savefile = f"{parameters['results_dir']}/" \
               f"{parameters['patient_idx']}/" \
               f"{parameters['slice_idx']}/" \
               f"{get_resname(parameters['resolutions'][parameters['iteration']])}/" \
               f"tform/ga_results.npy"

    ga_dict = dict()
    ga_dict["best_solution"] = best_solution
    ga_dict["solution_fitness"] = solution_fitness
    #np.save(savefile, ga_dict)

    # Plot some results
    plot_ga_result(quadrant_A, quadrant_B, quadrant_C, quadrant_D, best_solution)

    a=1

    return best_solution, solution_fitness


def fitness_func(solution, solution_idx):
    """
    Custom function to compute the cost related to the genetic algorithm. The genetic algorithm will provide
    a solution for the transformation matrix which can be used to compute the cost. A better transformation
    matrix should be associated with a lower cost. Cost should always be in the range [0, 2].
    """

    # General cost function costfun_parameters
    costfun_parameters["intensity_cost_weights"] = 1 - costfun_parameters["overunderlap_weights"][costfun_quadrant_A.iteration]
    costfun_parameters["center_of_mass"] = np.mean(costfun_parameters["image_centers"], axis=0)

    # Get transformation per quadrant from genetic algorithm solution
    ga_tform_A = solution[:3]
    ga_tform_B = solution[3:6]
    ga_tform_C = solution[6:9]
    ga_tform_D = solution[9:]

    """
    # Since a rotation with skimage always brings an additional translation, we need to recalculate the translation
    # by transforming the upper left bbox corner of each quadrant. This is also performed in the calculation of the initial
    # alignment.
    bbox_corners_a = [costfun_quadrant_A.bbox_corner_a, costfun_quadrant_A.bbox_corner_b,
                      costfun_quadrant_A.bbox_corner_c, costfun_quadrant_A.bbox_corner_d]
    bbox_corners_b = [costfun_quadrant_B.bbox_corner_a, costfun_quadrant_B.bbox_corner_b,
                      costfun_quadrant_B.bbox_corner_c, costfun_quadrant_B.bbox_corner_d]
    bbox_corners_c = [costfun_quadrant_C.bbox_corner_a, costfun_quadrant_C.bbox_corner_b,
                      costfun_quadrant_C.bbox_corner_c, costfun_quadrant_C.bbox_corner_d]
    bbox_corners_d = [costfun_quadrant_D.bbox_corner_a, costfun_quadrant_D.bbox_corner_b,
                      costfun_quadrant_D.bbox_corner_c, costfun_quadrant_D.bbox_corner_d]
    center_a_pre = np.mean(bbox_corners_a, axis=0)
    center_b_pre = np.mean(bbox_corners_b, axis=0)
    center_c_pre = np.mean(bbox_corners_c, axis=0)
    center_d_pre = np.mean(bbox_corners_d, axis=0)

    rot_tform_a = EuclideanTransform(rotation=-math.radians(ga_tform_A[2]), translation=(0, 0))
    rot_tform_b = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=(0, 0))
    rot_tform_c = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=(0, 0))
    rot_tform_d = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=(0, 0))

    rot_bbox_corners_a = np.squeeze(matrix_transform(bbox_corners_a, rot_tform_a.params))
    rot_bbox_corners_b = np.squeeze(matrix_transform(bbox_corners_b, rot_tform_b.params))
    rot_bbox_corners_c = np.squeeze(matrix_transform(bbox_corners_c, rot_tform_c.params))
    rot_bbox_corners_d = np.squeeze(matrix_transform(bbox_corners_d, rot_tform_d.params))

    center_a_post = np.mean(rot_bbox_corners_a, axis=0)
    center_b_post = np.mean(rot_bbox_corners_b, axis=0)
    center_c_post = np.mean(rot_bbox_corners_c, axis=0)
    center_d_post = np.mean(rot_bbox_corners_d, axis=0)

    trans_a = center_a_pre - center_a_post
    trans_b = center_b_pre - center_b_post
    trans_c = center_c_pre - center_c_post
    trans_d = center_d_pre - center_d_post

    final_trans_a = ga_tform_A[:2] + trans_a
    final_trans_b = ga_tform_B[:2] + trans_b
    final_trans_c = ga_tform_C[:2] + trans_c
    final_trans_d = ga_tform_D[:2] + trans_d

    final_tform_a = EuclideanTransform(rotation=-math.radians(ga_tform_A[2]), translation=final_trans_a)
    final_tform_b = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=final_trans_b)
    final_tform_c = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=final_trans_c)
    final_tform_d = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=final_trans_d)
    """
    final_tform_a = EuclideanTransform(rotation=-math.radians(ga_tform_A[2]), translation=ga_tform_A[:2])
    final_tform_b = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=ga_tform_B[:2])
    final_tform_c = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=ga_tform_C[:2])
    final_tform_d = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=ga_tform_D[:2])

    ## Rotate all edges and lines according to proposed transformation
    # Quadrant A
    costfun_quadrant_A.h_edge_tform = matrix_transform(costfun_quadrant_A.h_edge, final_tform_a.params)
    costfun_quadrant_A.v_edge_tform = matrix_transform(costfun_quadrant_A.v_edge, final_tform_a.params)
    costfun_quadrant_A.h_edge_theilsen_tform = matrix_transform(costfun_quadrant_A.h_edge_theilsen_coords, final_tform_a.params)
    costfun_quadrant_A.v_edge_theilsen_tform = matrix_transform(costfun_quadrant_A.v_edge_theilsen_coords, final_tform_a.params)

    # Quadrant B
    # edge1B_tform = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=(ga_tform_B[0], ga_tform_B[1]))
    costfun_quadrant_B.h_edge_tform = matrix_transform(costfun_quadrant_B.h_edge, final_tform_b.params)
    costfun_quadrant_B.v_edge_tform = matrix_transform(costfun_quadrant_B.v_edge, final_tform_b.params)
    costfun_quadrant_B.h_edge_theilsen_tform = matrix_transform(costfun_quadrant_B.h_edge_theilsen_coords, final_tform_b.params)
    costfun_quadrant_B.v_edge_theilsen_tform = matrix_transform(costfun_quadrant_B.v_edge_theilsen_coords, final_tform_b.params)

    # Horizontal edge quadrant C
    # edge1C_tform = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=(ga_tform_C[0], ga_tform_C[1]))
    costfun_quadrant_C.h_edge_tform = matrix_transform(costfun_quadrant_C.h_edge, final_tform_c.params)
    costfun_quadrant_C.v_edge_tform = matrix_transform(costfun_quadrant_C.v_edge, final_tform_c.params)
    costfun_quadrant_C.h_edge_theilsen_tform = matrix_transform(costfun_quadrant_C.h_edge_theilsen_coords, final_tform_c.params)
    costfun_quadrant_C.v_edge_theilsen_tform = matrix_transform(costfun_quadrant_C.v_edge_theilsen_coords, final_tform_c.params)

    # Horizontal edge quadrant C
    # edge1D_tform = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=(ga_tform_D[0], ga_tform_D[1]))
    costfun_quadrant_D.h_edge_tform = matrix_transform(costfun_quadrant_D.h_edge, final_tform_d.params)
    costfun_quadrant_D.v_edge_tform = matrix_transform(costfun_quadrant_D.v_edge, final_tform_d.params)
    costfun_quadrant_D.h_edge_theilsen_tform = matrix_transform(costfun_quadrant_D.h_edge_theilsen_coords, final_tform_d.params)
    costfun_quadrant_D.v_edge_theilsen_tform = matrix_transform(costfun_quadrant_D.v_edge_theilsen_coords, final_tform_d.params)

    # Packup some variables to loop over for computing global cost
    whichQuadrants = ['topBottom', 'topBottom', 'bottomTop', 'bottomTop',
                      'leftRight', 'rightLeft', 'leftRight', 'rightLeft']

    orientations = ["horizontal", "horizontal", "horizontal", "horizontal",
                    "vertical", "vertical", "vertical", "vertical"]

    edges_tform = [[costfun_quadrant_A.h_edge_tform, costfun_quadrant_C.h_edge_tform],
                   [costfun_quadrant_B.h_edge_tform, costfun_quadrant_D.h_edge_tform],
                   [costfun_quadrant_C.h_edge_tform, costfun_quadrant_A.h_edge_tform],
                   [costfun_quadrant_D.h_edge_tform, costfun_quadrant_B.h_edge_tform],
                   [costfun_quadrant_A.v_edge_tform, costfun_quadrant_B.v_edge_tform],
                   [costfun_quadrant_B.v_edge_tform, costfun_quadrant_A.v_edge_tform],
                   [costfun_quadrant_C.v_edge_tform, costfun_quadrant_D.v_edge_tform],
                   [costfun_quadrant_D.v_edge_tform, costfun_quadrant_C.v_edge_tform]]

    edges_theilsen_tform = [[costfun_quadrant_A.h_edge_theilsen_tform, costfun_quadrant_C.h_edge_theilsen_tform],
                            [costfun_quadrant_B.h_edge_theilsen_tform, costfun_quadrant_D.h_edge_theilsen_tform],
                            [costfun_quadrant_C.h_edge_theilsen_tform, costfun_quadrant_A.h_edge_theilsen_tform],
                            [costfun_quadrant_D.h_edge_theilsen_tform, costfun_quadrant_B.h_edge_theilsen_tform],
                            [costfun_quadrant_A.v_edge_theilsen_tform, costfun_quadrant_B.v_edge_theilsen_tform],
                            [costfun_quadrant_B.v_edge_theilsen_tform, costfun_quadrant_A.v_edge_theilsen_tform],
                            [costfun_quadrant_C.v_edge_theilsen_tform, costfun_quadrant_D.v_edge_theilsen_tform],
                            [costfun_quadrant_D.v_edge_theilsen_tform, costfun_quadrant_C.v_edge_theilsen_tform]]

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
                histograms[i][0], histograms[i][1], whichQuadrants[i], costfun_parameters, direction = orientations[i])

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
                intensities[i][0], intensities[i][1], whichQuadrants[i], costfun_parameters, direction = orientations[i])

            cost_result_dict["intensity_costs"].append(intensity_costs)
            cost_result_dict["overunderlap_costs"].append(overunderlap_costs)
            cost_result_dict["overhang_idxs"].append(overhang_idxs)
            cost_result_dict["edge_sample_idx"].append(edge_sample_idx)

            # Not sure why this variable is needed
            # intensityCosts[i][overhangIdxs[i]] = overhangPenalty

    else:
        raise ValueError(f"Unexpected cost function, options are: [simpleHists, rawIntensities]")

    # Compute mean and overall costs
    totalOverlapAndUnderlapCost = np.mean(cost_result_dict["overunderlap_costs"])
    totalIntensityCost = np.mean(cost_result_dict["intensity_costs"])

    cost = costfun_parameters["intensity_cost_weights"] * totalIntensityCost + \
           costfun_parameters["overunderlap_weights"][costfun_quadrant_A.iteration] * totalOverlapAndUnderlapCost

    allCosts = dict()
    allCosts["name"] = ["1A", "1B", "1C", "1D", "2A", "2B", "2C", "2D"]
    allCosts["intensityCost"] = totalOverlapAndUnderlapCost
    allCosts["overlapAndUnderlapCost"] = totalIntensityCost

    return 1/totalOverlapAndUnderlapCost
