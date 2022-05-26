import pygad
import numpy as np
import math
import copy

from skimage.transform import EuclideanTransform, matrix_transform, warp

from .plot_tools import *
from .get_resname import get_resname
from .get_cost_functions import get_cost_functions
from .recompute_transform import recompute_transform


def genetic_algorithm(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, initial_tform):
    """
    Function that runs a genetic algorithm using the pygad module.
    """

    # Make some variables global for fitness func
    global glob_quadrant_A, glob_quadrant_B, glob_quadrant_C, glob_quadrant_D
    global glob_parameters, glob_initialtform
    glob_quadrant_A = copy.deepcopy(quadrant_A)
    glob_quadrant_B = copy.deepcopy(quadrant_B)
    glob_quadrant_C = copy.deepcopy(quadrant_C)
    glob_quadrant_D = copy.deepcopy(quadrant_D)
    glob_parameters = copy.deepcopy(parameters)
    glob_initialtform = copy.deepcopy(initial_tform)

    # Initialize parameters
    tform_combi = [*initial_tform[quadrant_A.quadrant_name][:-1], *initial_tform[quadrant_B.quadrant_name][:-1],
                   *initial_tform[quadrant_C.quadrant_name][:-1], *initial_tform[quadrant_D.quadrant_name][:-1]]
    num_genes = len(tform_combi)
    ga_tform = np.zeros((num_genes))
    init_fitness = fitness_func(ga_tform, 0)
    num_sol = 20
    num_gen = 100
    keep_parents = 5

    # Cap solution ranges to plausible values
    t_range = parameters["translation_ranges"][parameters["iteration"]]
    a_range = parameters["angle_range"][parameters["iteration"]]
    angles = [False, False, True]*4
    lb = [int(x - a_range) if is_angle else int(x - t_range) for x, is_angle in zip(ga_tform, angles)]
    ub = [int(x + a_range) if is_angle else int(x + t_range) for x, is_angle in zip(ga_tform, angles)]
    param_range = [[l, u] for l, u in zip(lb, ub)]

    # Generate initial population based on noise
    init_pop = np.zeros((num_sol, num_genes))
    for i in range(num_sol):
        np.random.seed(i)
        translation_noise = np.random.randint(low=-t_range/10, high=t_range/10, size=num_genes-4)
        angle_noise = np.random.randint(low=-a_range/5, high=a_range/5, size=4)
        total_noise = [*translation_noise[:2], angle_noise[0], *translation_noise[2:4], angle_noise[1],
                       *translation_noise[4:6], angle_noise[2], *translation_noise[6:], angle_noise[3]]
        init_pop[i, :] = ga_tform + total_noise

    # Pygad has a wide variety of parameters for the optimization. Parents with a (M) were copied from the Matlab
    # implementation. Other parameters are chosen empirically.
    ga_instance = pygad.GA(
        num_generations = num_gen,              # num generations to optimize
        fitness_func=fitness_func,              # optimization function
        initial_population=init_pop,            # values for first-gen genes
        gene_space=param_range,                 # (M) parameter search range
        keep_parents = keep_parents,            # (M) num of parents that proceed to next generation unaltered
        num_parents_mating = 10,                # num parents that produce offspring
        parent_selection_type="rank",           # function to select parents for offspring
        crossover_type="scattered",             # (M) gene selection
        crossover_probability=0.5,              # probability that a parent is chosen for crossover
        mutation_type="random",                 # mutation type
        mutation_probability=0.2,               # probability that a gene mutates
        random_mutation_min_val=-3,             # lower bound and upper bound for a value that
        random_mutation_max_val=3,              #   is added to the gene
        gene_type = [float, 1],
        save_best_solutions=True,
        suppress_warnings=True,
        stop_criteria=None                      # this could later be set to stop after loss plateaus
    )

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

    if parameters["iteration"] == 0:
        parameters["GA_fitness"].append(init_fitness)

    parameters["GA_fitness"].append(solution_fitness)

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

    # General cost function glob_parameters
    glob_parameters["intensity_cost_weights"] = 1 - glob_parameters["overunderlap_weights"][glob_quadrant_A.iteration]
    glob_parameters["center_of_mass"] = np.mean(glob_parameters["image_centers"], axis=0)
    eps = 1e-5

    # Recompute transform to account for translation induced by rotation
    tform_a, tform_b, tform_c, tform_d = recompute_transform(glob_quadrant_A, glob_quadrant_B,
                                                             glob_quadrant_C, glob_quadrant_D,
                                                             solution)

    # Make tform matrices
    final_tform_a = EuclideanTransform(rotation=-math.radians(tform_a[2]), translation=tform_a[:2])
    final_tform_b = EuclideanTransform(rotation=-math.radians(tform_b[2]), translation=tform_b[:2])
    final_tform_c = EuclideanTransform(rotation=-math.radians(tform_c[2]), translation=tform_c[:2])
    final_tform_d = EuclideanTransform(rotation=-math.radians(tform_d[2]), translation=tform_d[:2])

    ## Rotate all edges and lines according to proposed transformation
    # Quadrant A
    glob_quadrant_A.h_edge_tform = matrix_transform(glob_quadrant_A.h_edge, final_tform_a.params)
    glob_quadrant_A.v_edge_tform = matrix_transform(glob_quadrant_A.v_edge, final_tform_a.params)
    glob_quadrant_A.h_edge_theilsen_tform = matrix_transform(glob_quadrant_A.h_edge_theilsen_coords, final_tform_a.params)
    glob_quadrant_A.v_edge_theilsen_tform = matrix_transform(glob_quadrant_A.v_edge_theilsen_coords, final_tform_a.params)
    glob_quadrant_A.h_edge_theilsen_tform = matrix_transform(glob_quadrant_A.h_edge_theilsen_endpoints, final_tform_a.params)
    glob_quadrant_A.v_edge_theilsen_tform = matrix_transform(glob_quadrant_A.v_edge_theilsen_coords, final_tform_a.params)

    # Quadrant B
    # edge1B_tform = EuclideanTransform(rotation=-math.radians(ga_tform_B[2]), translation=(ga_tform_B[0], ga_tform_B[1]))
    glob_quadrant_B.h_edge_tform = matrix_transform(glob_quadrant_B.h_edge, final_tform_b.params)
    glob_quadrant_B.v_edge_tform = matrix_transform(glob_quadrant_B.v_edge, final_tform_b.params)
    glob_quadrant_B.h_edge_theilsen_tform = matrix_transform(glob_quadrant_B.h_edge_theilsen_coords, final_tform_b.params)
    glob_quadrant_B.v_edge_theilsen_tform = matrix_transform(glob_quadrant_B.v_edge_theilsen_coords, final_tform_b.params)

    # Horizontal edge quadrant C
    # edge1C_tform = EuclideanTransform(rotation=-math.radians(ga_tform_C[2]), translation=(ga_tform_C[0], ga_tform_C[1]))
    glob_quadrant_C.h_edge_tform = matrix_transform(glob_quadrant_C.h_edge, final_tform_c.params)
    glob_quadrant_C.v_edge_tform = matrix_transform(glob_quadrant_C.v_edge, final_tform_c.params)
    glob_quadrant_C.h_edge_theilsen_tform = matrix_transform(glob_quadrant_C.h_edge_theilsen_coords, final_tform_c.params)
    glob_quadrant_C.v_edge_theilsen_tform = matrix_transform(glob_quadrant_C.v_edge_theilsen_coords, final_tform_c.params)

    # Horizontal edge quadrant C
    # edge1D_tform = EuclideanTransform(rotation=-math.radians(ga_tform_D[2]), translation=(ga_tform_D[0], ga_tform_D[1]))
    glob_quadrant_D.h_edge_tform = matrix_transform(glob_quadrant_D.h_edge, final_tform_d.params)
    glob_quadrant_D.v_edge_tform = matrix_transform(glob_quadrant_D.v_edge, final_tform_d.params)
    glob_quadrant_D.h_edge_theilsen_tform = matrix_transform(glob_quadrant_D.h_edge_theilsen_coords, final_tform_d.params)
    glob_quadrant_D.v_edge_theilsen_tform = matrix_transform(glob_quadrant_D.v_edge_theilsen_coords, final_tform_d.params)

    # Get tformed images
    A_tform_image = warp(glob_quadrant_A.tform_image, final_tform_a.inverse)
    B_tform_image = warp(glob_quadrant_B.tform_image, final_tform_b.inverse)
    C_tform_image = warp(glob_quadrant_C.tform_image, final_tform_c.inverse)
    D_tform_image = warp(glob_quadrant_D.tform_image, final_tform_d.inverse)

    ### COST: OVERLAP BETWEEN QUADRANTS ###
    # Compute relative overlap in image
    total_size = np.sum(A_tform_image>0) + np.sum(B_tform_image>0) + np.sum(C_tform_image>0) + np.sum(D_tform_image>0)
    combined_image = (A_tform_image>0)*1 + (B_tform_image>0)*1 + (C_tform_image>0)*1 + (D_tform_image>0)*1
    absolute_overlap = np.sum(combined_image>1)
    relative_overlap = absolute_overlap/total_size
    overlap_costs = glob_parameters["overlap_weight"] * relative_overlap
    ###


    # Packup some variables to loop over for computing global cost
    whichQuadrants = ['topBottom', 'topBottom', 'bottomTop', 'bottomTop',
                      'leftRight', 'rightLeft', 'leftRight', 'rightLeft']

    orientations = ["horizontal", "horizontal", "horizontal", "horizontal",
                    "vertical", "vertical", "vertical", "vertical"]

    edges_tform = [[glob_quadrant_A.h_edge_tform, glob_quadrant_C.h_edge_tform],
                   [glob_quadrant_B.h_edge_tform, glob_quadrant_D.h_edge_tform],
                   [glob_quadrant_C.h_edge_tform, glob_quadrant_A.h_edge_tform],
                   [glob_quadrant_D.h_edge_tform, glob_quadrant_B.h_edge_tform],
                   [glob_quadrant_A.v_edge_tform, glob_quadrant_B.v_edge_tform],
                   [glob_quadrant_B.v_edge_tform, glob_quadrant_A.v_edge_tform],
                   [glob_quadrant_C.v_edge_tform, glob_quadrant_D.v_edge_tform],
                   [glob_quadrant_D.v_edge_tform, glob_quadrant_C.v_edge_tform]]

    edges_theilsen_tform = [[glob_quadrant_A.h_edge_theilsen_tform, glob_quadrant_C.h_edge_theilsen_tform],
                            [glob_quadrant_B.h_edge_theilsen_tform, glob_quadrant_D.h_edge_theilsen_tform],
                            [glob_quadrant_C.h_edge_theilsen_tform, glob_quadrant_A.h_edge_theilsen_tform],
                            [glob_quadrant_D.h_edge_theilsen_tform, glob_quadrant_B.h_edge_theilsen_tform],
                            [glob_quadrant_A.v_edge_theilsen_tform, glob_quadrant_B.v_edge_theilsen_tform],
                            [glob_quadrant_B.v_edge_theilsen_tform, glob_quadrant_A.v_edge_theilsen_tform],
                            [glob_quadrant_C.v_edge_theilsen_tform, glob_quadrant_D.v_edge_theilsen_tform],
                            [glob_quadrant_D.v_edge_theilsen_tform, glob_quadrant_C.v_edge_theilsen_tform]]

    """
    intensities = [[glob_quadrant_A.intensities_h, glob_quadrant_C.intensities_h],
                   [glob_quadrant_B.intensities_h, glob_quadrant_D.intensities_h],
                   [glob_quadrant_C.intensities_h, glob_quadrant_A.intensities_h],
                   [glob_quadrant_D.intensities_h, glob_quadrant_B.intensities_h],
                   [glob_quadrant_A.intensities_v, glob_quadrant_B.intensities_v],
                   [glob_quadrant_B.intensities_v, glob_quadrant_A.intensities_v],
                   [glob_quadrant_C.intensities_v, glob_quadrant_D.intensities_v],
                   [glob_quadrant_D.intensities_v, glob_quadrant_C.intensities_v]]

    histograms = [[glob_quadrant_A.hists_h, glob_quadrant_C.hists_h],
                  [glob_quadrant_B.hists_h, glob_quadrant_D.hists_h],
                  [glob_quadrant_C.hists_h, glob_quadrant_A.hists_h],
                  [glob_quadrant_D.hists_h, glob_quadrant_B.hists_h],
                  [glob_quadrant_A.hists_v, glob_quadrant_B.hists_v],
                  [glob_quadrant_B.hists_v, glob_quadrant_A.hists_v],
                  [glob_quadrant_C.hists_v, glob_quadrant_D.hists_v],
                  [glob_quadrant_D.hists_v, glob_quadrant_C.hists_v]]
    """

    # Preallocate dict for saving results
    cost_result_dict = dict()
    cost_result_dict["intensity_costs"] = []
    cost_result_dict["overunderlap_costs"] = []
    dict_intensity_costs = {}

    histograms = np.zeros((8, 2))
    intensities = np.zeros((8, 2))

    ### COST: DISTANCE BETWEEN EDGE POINTS OF THEILSEN LINES ###
    # Histogram-based cost function
    if glob_parameters["cost_functions"][glob_quadrant_A.iteration] == "simple_hists":

        for i in range(8):
            intensity_costs, overunderlap_costs = get_cost_functions(
                edges_tform[i][0],
                edges_tform[i][1],
                edges_theilsen_tform[i][0],
                edges_theilsen_tform[i][1],
                histograms[i][0],
                histograms[i][1],
                whichQuadrants[i],
                glob_parameters,
                direction = orientations[i],
                quadrant_shape=np.shape(glob_quadrant_A.tform_image)
            )

            cost_result_dict["intensity_costs"].append(intensity_costs + eps)
            cost_result_dict["overunderlap_costs"].append(overunderlap_costs + eps)

    # Intensity-based cost function
    elif glob_parameters["cost_functions"][glob_quadrant_A.iteration] == "raw_intensities":

        for i in range(8):
            intensity_costs, overunderlap_costs = get_cost_functions(
                edges_tform[i][0],
                edges_tform[i][1],
                edges_theilsen_tform[i][0],
                edges_theilsen_tform[i][1],
                intensities[i][0],
                intensities[i][1],
                whichQuadrants[i],
                glob_parameters,
                direction = orientations[i],
                quadrant_shape=np.shape(glob_quadrant_A.tform_image)
            )

            cost_result_dict["intensity_costs"].append(intensity_costs)
            cost_result_dict["overunderlap_costs"].append(overunderlap_costs)

    else:
        raise ValueError(f"Unexpected cost function, options are: [simpleHists, rawIntensities]")

    # Compute mean and overall costs
    total_overunderlap_costs = np.mean(cost_result_dict["overunderlap_costs"])
    #total_intensity_costs = np.mean(cost_result_dict["intensity_costs"])

    return 1/(total_overunderlap_costs+overlap_costs)

