import pygad
import numpy as np
import os
import math
import copy

from skimage.transform import EuclideanTransform, matrix_transform

from .plot_tools import *
from .get_resname import get_resname
from .rotate_cp import rotate_cp, rotate_2dpoints
from .get_cost_functions import get_cost_functions
from .recompute_transform import recompute_transform


def genetic_algorithm(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, initial_tform):
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
    num_genes = len(tform_combi)
    ga_tform = np.zeros((num_genes))
    init_fitness = fitness_func(ga_tform, 0)

    # Cap solution ranges to plausible values
    t_range = parameters["translation_ranges"][parameters["iteration"]]
    a_range = parameters["angle_range"]
    angles = [False, False, True]*4
    lb = [x - a_range if is_angle else x - t_range for x, is_angle in zip(ga_tform, angles)]
    ub = [x + a_range if is_angle else x + t_range for x, is_angle in zip(ga_tform, angles)]
    param_range = [[l, u] for l, u in zip(lb, ub)]

    # Define values for initial population. These should be relatively close to initial tform
    num_sol = 20
    num_gen = 50
    init_pop = np.zeros((num_sol, num_genes))

    # Generate initial population based on noise
    for i in range(num_sol):
        np.random.seed(i)
        translation_noise = np.random.randint(low=-t_range, high=t_range, size=num_genes-4)/10
        angle_noise = np.random.randint(low=-a_range, high=a_range, size=4)/a_range
        total_noise = [*translation_noise[:2], angle_noise[0], *translation_noise[2:4], angle_noise[1],
                       *translation_noise[4:6], angle_noise[2], *translation_noise[6:], angle_noise[3]]
        init_pop[i, :] = ga_tform + total_noise

    # Pygad has a wide variety of parameters for the optimization. Parents with a (M) were copied from the Matlab
    # implementation. Other parameters are chosen empirically.
    ga_instance = pygad.GA(
        num_generations = num_gen,              # num generations to optimize
        keep_parents=int(0.05*num_sol),         # (M) num of parents that proceed to next generation unaltered
        num_parents_mating = 5,                 # num solutions to keep per generation
        parent_selection_type="sus",            # (M)
        fitness_func = fitness_func,            # optimization function
        initial_population = init_pop,          # values for first-gen genes
        gene_space = param_range,               # (M) parameter search range
        mutation_probability=0.3,               # probability that a gene mutates
        mutation_type="random",                 # mutation type
        random_mutation_min_val=-5,             # max value
        random_mutation_max_val=5,
        crossover_type="scattered",             # (M) gene selection
        save_best_solutions=True,
        suppress_warnings=True,
        stop_criteria=None                      # this could later be set to stop after loss plateaus
    )

    # Run and show results
    ga_instance.run()
    ga_instance.plot_fitness()

    # Retrieve best results and save
    best_solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # The provided solution does not account for translation induced by rotation. The recompute transform
    # function will update the transformation by including the translation in the tform matrix.
    solution = recompute_transform(quadrant_A, quadrant_B, quadrant_C, quadrant_D, best_solution)

    # Save the solution results
    savefile = f"{parameters['results_dir']}/" \
               f"{parameters['patient_idx']}/" \
               f"{parameters['slice_idx']}/" \
               f"{get_resname(parameters['resolutions'][parameters['iteration']])}/" \
               f"tform/tform_ga.npy"

    ga_dict = dict()
    ga_dict["best_solution"] = solution
    ga_dict["solution_fitness"] = solution_fitness
    ga_dict["initial_fitness"] = init_fitness

    return ga_dict


def fitness_func(solution, solution_idx):
    """
    Custom function to compute the cost related to the genetic algorithm. The genetic algorithm will provide
    a solution for the transformation matrix which can be used to compute the cost. A better transformation
    matrix should be associated with a lower cost. Cost should always be in the range [0, 2].
    """

    # General cost function costfun_parameters
    costfun_parameters["intensity_cost_weights"] = 1 - costfun_parameters["overunderlap_weights"][costfun_quadrant_A.iteration]
    costfun_parameters["center_of_mass"] = np.mean(costfun_parameters["image_centers"], axis=0)
    eps = 1e-5

    # Recompute transform to account for translation induced by rotation
    tform_a, tform_b, tform_c, tform_d = recompute_transform(costfun_quadrant_A, costfun_quadrant_B,
                                                             costfun_quadrant_C, costfun_quadrant_D,
                                                             solution)

    # Make tform matrices
    final_tform_a = EuclideanTransform(rotation=-math.radians(tform_a[2]), translation=tform_a[:2])
    final_tform_b = EuclideanTransform(rotation=-math.radians(tform_b[2]), translation=tform_b[:2])
    final_tform_c = EuclideanTransform(rotation=-math.radians(tform_c[2]), translation=tform_c[:2])
    final_tform_d = EuclideanTransform(rotation=-math.radians(tform_d[2]), translation=tform_d[:2])

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

    """
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
    """

    # Preallocate dict for saving results
    cost_result_dict = dict()
    cost_result_dict["intensity_costs"] = []
    cost_result_dict["overunderlap_costs"] = []

    dict_intensity_costs = {}

    histograms = np.zeros((8, 2))
    intensities = np.zeros((8, 2))

    # Histogram-based cost function
    if costfun_parameters["cost_functions"][costfun_quadrant_A.iteration] == "simple_hists":

        for i in range(8):
            intensity_costs, overunderlap_costs = get_cost_functions(
                edges_tform[i][0], edges_tform[i][1], edges_theilsen_tform[i][0], edges_theilsen_tform[i][1],
                histograms[i][0], histograms[i][1], whichQuadrants[i], costfun_parameters, direction = orientations[i])

            #cost_result_dict["intensity_costs"].append(intensity_costs + eps)
            cost_result_dict["overunderlap_costs"].append(overunderlap_costs + eps)

            # Not sure why this variable is needed
            # intensityCosts[i][overhangIdxs[i]] = overhangPenalty

    # Intensity-based cost function
    elif costfun_parameters["cost_functions"][costfun_quadrant_A.iteration] == "raw_intensities":

        for i in range(8):
            intensity_costs, overunderlap_costs = get_cost_functions(
                edges_tform[i][0], edges_tform[i][1], edges_theilsen_tform[i][0], edges_theilsen_tform[i][1],
                intensities[i][0], intensities[i][1], whichQuadrants[i], costfun_parameters, direction = orientations[i])

            cost_result_dict["intensity_costs"].append(intensity_costs + eps)
            cost_result_dict["overunderlap_costs"].append(overunderlap_costs + eps)

            # Not sure why this variable is needed
            # intensityCosts[i][overhangIdxs[i]] = overhangPenalty

    else:
        raise ValueError(f"Unexpected cost function, options are: [simpleHists, rawIntensities]")

    # Compute mean and overall costs
    totalOverlapAndUnderlapCost = np.mean(cost_result_dict["overunderlap_costs"])
    #totalIntensityCost = np.mean(cost_result_dict["intensity_costs"])

    """
    cost = costfun_parameters["intensity_cost_weights"] * totalIntensityCost + \
           costfun_parameters["overunderlap_weights"][costfun_quadrant_A.iteration] * totalOverlapAndUnderlapCost

    allCosts = dict()
    allCosts["name"] = ["1A", "1B", "1C", "1D", "2A", "2B", "2C", "2D"]
    allCosts["intensityCost"] = totalIntensityCost
    allCosts["overlapAndUnderlapCost"] = totalOverlapAndUnderlapCost
    """

    return 1/totalOverlapAndUnderlapCost
