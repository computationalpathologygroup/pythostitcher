import pygad
import numpy as np
import copy

from .plot_tools import *
from .get_cost_functions import *
from .recompute_transform import *


def genetic_algorithm(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, initial_tform):
    """
    Function that runs a genetic algorithm using the pygad module. This function will use a global assembly method
    where the stitching for all quadrants is optimised at once. This is in contrast with the other piecewise
    method where quadrants are stitched one by one.

    Input:
    - all quadrants
    - dict with parameters
    - initial transformation matrix

    Output:
    - dict with optimized transformation matrix and the fitness of that matrix
    """

    # Make some variables global for access by fitness function. The fitness function is built to not accept any
    # other input parameters than the solution and solution index, hence using global is necessary.
    global glob_quadrant_A, glob_quadrant_B, glob_quadrant_C, glob_quadrant_D, glob_parameters, num_gen, glob_init_tform
    glob_quadrant_A = copy.deepcopy(quadrant_A)
    glob_quadrant_B = copy.deepcopy(quadrant_B)
    glob_quadrant_C = copy.deepcopy(quadrant_C)
    glob_quadrant_D = copy.deepcopy(quadrant_D)
    glob_parameters = copy.deepcopy(parameters)
    glob_init_tform = copy.deepcopy(initial_tform)

    # Initialize parameters
    tform_combi = [*initial_tform[quadrant_A.quadrant_name][:-2], *initial_tform[quadrant_B.quadrant_name][:-2],
                   *initial_tform[quadrant_C.quadrant_name][:-2], *initial_tform[quadrant_D.quadrant_name][:-2]]
    num_genes = len(tform_combi)
    ga_tform = np.zeros((num_genes))
    glob_parameters["distance_scaling_required"] = True
    init_fitness = fitness_func(ga_tform, 0)
    num_sol = parameters["n_solutions"]
    num_gen = parameters["n_generations"]
    keep_parents = parameters["n_parents"]
    parents_mating = parameters["n_mating"]
    parents = parameters["parent_selection"]
    p_crossover = parameters["p_crossover"]
    crossover_type = parameters["crossover_type"]
    p_mutation = parameters["p_mutation"]
    mutation_type = parameters["mutation_type"]

    # Cap solution ranges to plausible values
    t_range = parameters["translation_range"][parameters["iteration"]] * int(np.mean(quadrant_A.tform_image.shape))
    a_range = parameters["angle_range"][parameters["iteration"]]
    angles = [False, False, True]*4
    lb = [int(x - a_range) if is_angle else int(x - t_range) for x, is_angle in zip(ga_tform, angles)]
    ub = [int(x + a_range) if is_angle else int(x + t_range) for x, is_angle in zip(ga_tform, angles)]
    param_range = [np.arange(l, u, step=0.1) if a else np.arange(l, u, step=1) for l, u, a in zip(lb, ub, angles)]

    # Generate initial population based on noise
    init_pop = np.zeros((num_sol, num_genes))
    for i in range(num_sol):
        np.random.seed(i)
        translation_noise = np.random.randint(low=np.round(-t_range), high=np.round(t_range), size=num_genes-4)
        angle_noise = np.random.randint(low=-a_range/5, high=a_range/5, size=4)
        total_noise = [*translation_noise[:2], angle_noise[0], *translation_noise[2:4], angle_noise[1],
                       *translation_noise[4:6], angle_noise[2], *translation_noise[6:], angle_noise[3]]
        init_pop[i, :] = ga_tform + total_noise

    # Pygad has a wide variety of parameters for the optimization. Parents with a (M) were copied from the Matlab
    # implementation. Other parameters are chosen empirically.
    ga_instance = pygad.GA(
        num_generations = num_gen,                  # num generations to optimize
        fitness_func=fitness_func,                  # optimization function
        initial_population=init_pop,                # values for first-gen genes
        gene_space=param_range,                     # parameter search range
        keep_parents = keep_parents,                # num of parents that proceed to next generation unaltered
        num_parents_mating = parents_mating,        # num parents that produce offspring
        parent_selection_type=parents,              # function to select parents for offspring
        crossover_type=crossover_type,              # (M) gene selection
        crossover_probability=p_crossover,          # probability that a parent is chosen for crossover
        mutation_type=mutation_type,                # mutation type
        mutation_probability=p_mutation,            # probability that a gene mutates
        on_generation=plot_best_sol_per_gen,        # plot the result after every X (currently 10) generations
        save_best_solutions=True,                   # must be True in order to use the callback above
        suppress_warnings=True,                     # suppress annoying and irrelevant warnings
        stop_criteria=None                          # this could later be set to stop after loss plateaus
    )

    # Run genetic algorithm
    ga_instance.run()
    ga_instance.plot_fitness()

    # Retrieve best results and save
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Get solution per quadrant
    sol_A = [*solution[:3], glob_quadrant_A.image_center_peri, glob_init_tform[glob_quadrant_A.quadrant_name][4]]
    sol_B = [*solution[3:6], glob_quadrant_B.image_center_peri, glob_init_tform[glob_quadrant_B.quadrant_name][4]]
    sol_C = [*solution[6:9], glob_quadrant_C.image_center_peri, glob_init_tform[glob_quadrant_C.quadrant_name][4]]
    sol_D = [*solution[9:], glob_quadrant_D.image_center_peri, glob_init_tform[glob_quadrant_D.quadrant_name][4]]

    # Combine solution with initial transformation
    combo_tform_a = [
        *[a + b for a, b in zip(glob_init_tform[glob_quadrant_A.quadrant_name][:3], sol_A[:3])],
        glob_init_tform[glob_quadrant_A.quadrant_name][3], glob_init_tform[glob_quadrant_A.quadrant_name][4]
    ]
    combo_tform_b = [
        *[a + b for a, b in zip(glob_init_tform[glob_quadrant_B.quadrant_name][:3], sol_B[:3])],
        glob_init_tform[glob_quadrant_B.quadrant_name][3], glob_init_tform[glob_quadrant_B.quadrant_name][4]
    ]
    combo_tform_c = [
        *[a + b for a, b in zip(glob_init_tform[glob_quadrant_C.quadrant_name][:3], sol_C[:3])],
        glob_init_tform[glob_quadrant_C.quadrant_name][3], glob_init_tform[glob_quadrant_C.quadrant_name][4]
    ]
    combo_tform_d = [
        *[a + b for a, b in zip(glob_init_tform[glob_quadrant_D.quadrant_name][:3], sol_D[:3])],
        glob_init_tform[glob_quadrant_D.quadrant_name][3], glob_init_tform[glob_quadrant_D.quadrant_name][4]
    ]

    # Save some results for later plotting
    if parameters["iteration"] == 0:
        parameters["GA_fitness"].append(init_fitness)
    parameters["GA_fitness"].append(solution_fitness)

    ga_dict = dict()
    ga_dict["fitness"] = solution_fitness
    tforms = [combo_tform_a, combo_tform_b, combo_tform_c, combo_tform_d]
    for t, key in zip(tforms, parameters["filenames"].keys()):
        ga_dict[key] = t

    return ga_dict


def fitness_func(solution, solution_idx):
    """
    Custom function to compute the cost related to the genetic algorithm. The genetic algorithm will provide
    a solution for the transformation matrix which can be used to compute the cost. A better transformation
    matrix should be associated with a lower cost and thus a higher fitness.

    Although histogram matching and overlap are currently supported as cost functions, these require warping the
    image rather than some coordinates and are therefore much more computationally expensive.
    """

    global glob_quadrant_A, glob_quadrant_B, glob_quadrant_C, glob_quadrant_D, glob_parameters

    # General cost function glob_parameters
    glob_parameters["center_of_mass"] = np.mean(glob_parameters["image_centers"], axis=0)
    eps = 1e-5

    # Get solution per quadrant
    sol_A = [*solution[:3], glob_quadrant_A.image_center_peri, glob_init_tform[glob_quadrant_A.quadrant_name][4]]
    sol_B = [*solution[3:6], glob_quadrant_B.image_center_peri, glob_init_tform[glob_quadrant_B.quadrant_name][4]]
    sol_C = [*solution[6:9], glob_quadrant_C.image_center_peri, glob_init_tform[glob_quadrant_C.quadrant_name][4]]
    sol_D = [*solution[9:], glob_quadrant_D.image_center_peri, glob_init_tform[glob_quadrant_D.quadrant_name][4]]

    # Combine solution with initial transformation
    combo_tform_a = [
        *[a + b for a, b in zip(glob_init_tform[glob_quadrant_A.quadrant_name][:3], sol_A[:3])],
        glob_init_tform[glob_quadrant_A.quadrant_name][3], glob_init_tform[glob_quadrant_A.quadrant_name][4]
    ]
    combo_tform_b = [
        *[a + b for a, b in zip(glob_init_tform[glob_quadrant_B.quadrant_name][:3], sol_B[:3])],
            glob_init_tform[glob_quadrant_B.quadrant_name][3], glob_init_tform[glob_quadrant_B.quadrant_name][4]
    ]
    combo_tform_c = [
        *[a + b for a, b in zip(glob_init_tform[glob_quadrant_C.quadrant_name][:3], sol_C[:3])],
        glob_init_tform[glob_quadrant_C.quadrant_name][3], glob_init_tform[glob_quadrant_C.quadrant_name][4]
    ]
    combo_tform_d = [
        *[a + b for a, b in zip(glob_init_tform[glob_quadrant_D.quadrant_name][:3], sol_D[:3])],
        glob_init_tform[glob_quadrant_D.quadrant_name][3], glob_init_tform[glob_quadrant_D.quadrant_name][4]
    ]

    # Apply tform to several attributes of the quadrant
    glob_quadrant_A = apply_new_transform(glob_quadrant_A, sol_A, combo_tform_a, tform_image=False)
    glob_quadrant_B = apply_new_transform(glob_quadrant_B, sol_B, combo_tform_b, tform_image=False)
    glob_quadrant_C = apply_new_transform(glob_quadrant_C, sol_C, combo_tform_c, tform_image=False)
    glob_quadrant_D = apply_new_transform(glob_quadrant_D, sol_D, combo_tform_d, tform_image=False)

    """
    histogram_cost = hist_cost_function(glob_quadrant_A, glob_quadrant_B,
                                            glob_quadrant_C, glob_quadrant_D,
                                            glob_parameters, plot=False)
    
    overlap_cost = overlap_cost_function(glob_quadrant_A, glob_quadrant_B,
                             glob_quadrant_C, glob_quadrant_D,
                             tforms=[tform_a, tform_b, tform_c, tform_d])
    
    """

    distance_cost = distance_cost_function(glob_quadrant_A, glob_quadrant_B,
                                           glob_quadrant_C, glob_quadrant_D,
                                           glob_parameters)

    #total_cost = histogram_cost + overlap_cost + distance_cost

    return 1 / (distance_cost)


def plot_best_sol_per_gen(ga):
    """
    Custom function to plot the result of the best solution for each generation.
    """
    global glob_quadrant_A, glob_quadrant_B, glob_quadrant_C, glob_quadrant_D, num_gen

    # Get best solution and process it
    solution, fitness, _ = ga.best_solution()
    gen = ga.generations_completed
    solution_dict = dict()

    # Get solution per quadrant
    sol_A = [*solution[:3], glob_quadrant_A.image_center_peri, glob_init_tform[glob_quadrant_A.quadrant_name][4]]
    sol_B = [*solution[3:6], glob_quadrant_B.image_center_peri, glob_init_tform[glob_quadrant_B.quadrant_name][4]]
    sol_C = [*solution[6:9], glob_quadrant_C.image_center_peri, glob_init_tform[glob_quadrant_C.quadrant_name][4]]
    sol_D = [*solution[9:], glob_quadrant_D.image_center_peri, glob_init_tform[glob_quadrant_D.quadrant_name][4]]

    # Show solution for every N generations
    if gen % 50 == 0:
        # Combine solution with initial transformation
        combo_tform_a = [
            *[a + b for a, b in zip(glob_init_tform[glob_quadrant_A.quadrant_name][:3], sol_A[:3])],
            glob_init_tform[glob_quadrant_A.quadrant_name][3], glob_init_tform[glob_quadrant_A.quadrant_name][4]
        ]
        combo_tform_b = [
            *[a + b for a, b in zip(glob_init_tform[glob_quadrant_B.quadrant_name][:3], sol_B[:3])],
            glob_init_tform[glob_quadrant_B.quadrant_name][3], glob_init_tform[glob_quadrant_B.quadrant_name][4]
        ]
        combo_tform_c = [
            *[a + b for a, b in zip(glob_init_tform[glob_quadrant_C.quadrant_name][:3], sol_C[:3])],
            glob_init_tform[glob_quadrant_C.quadrant_name][3], glob_init_tform[glob_quadrant_C.quadrant_name][4]
        ]
        combo_tform_d = [
            *[a + b for a, b in zip(glob_init_tform[glob_quadrant_D.quadrant_name][:3], sol_D[:3])],
            glob_init_tform[glob_quadrant_D.quadrant_name][3], glob_init_tform[glob_quadrant_D.quadrant_name][4]
        ]

        # Apply tform to several attributes of the quadrant
        glob_quadrant_A = apply_new_transform(glob_quadrant_A, sol_A, combo_tform_a, tform_image=True)
        glob_quadrant_B = apply_new_transform(glob_quadrant_B, sol_B, combo_tform_b, tform_image=True)
        glob_quadrant_C = apply_new_transform(glob_quadrant_C, sol_C, combo_tform_c, tform_image=True)
        glob_quadrant_D = apply_new_transform(glob_quadrant_D, sol_D, combo_tform_d, tform_image=True)

        # Get final image
        total_im = recombine_quadrants(glob_quadrant_A.colour_image, glob_quadrant_B.colour_image,
                                       glob_quadrant_C.colour_image, glob_quadrant_D.colour_image)

        # Plotting parameters
        ratio = glob_parameters["resolutions"][glob_parameters["iteration"]] / glob_parameters["resolutions"][0]
        ms = np.sqrt(2500 * np.sqrt(ratio))

        # Make figure
        plt.figure()
        plt.title(f"Best result at generation {gen}/{num_gen}: fitness = {np.round(fitness, 2)}")
        plt.imshow(total_im, cmap="gray")
        plt.scatter(glob_quadrant_A.v_edge_theilsen_endpoints_tform[:, 0],
                    glob_quadrant_A.v_edge_theilsen_endpoints_tform[:, 1],
                    marker='*', s=ms, color="g")
        plt.scatter(glob_quadrant_A.h_edge_theilsen_endpoints_tform[:, 0],
                    glob_quadrant_A.h_edge_theilsen_endpoints_tform[:, 1],
                    marker='+', s=ms, color="b")
        plt.scatter(glob_quadrant_B.v_edge_theilsen_endpoints_tform[:, 0],
                    glob_quadrant_B.v_edge_theilsen_endpoints_tform[:, 1],
                    marker='*', s=ms, color="g")
        plt.scatter(glob_quadrant_B.h_edge_theilsen_endpoints_tform[:, 0],
                    glob_quadrant_B.h_edge_theilsen_endpoints_tform[:, 1],
                    marker='+', s=ms, color="b")
        plt.scatter(glob_quadrant_C.v_edge_theilsen_endpoints_tform[:, 0],
                    glob_quadrant_C.v_edge_theilsen_endpoints_tform[:, 1],
                    marker='*', s=ms, color="g")
        plt.scatter(glob_quadrant_C.h_edge_theilsen_endpoints_tform[:, 0],
                    glob_quadrant_C.h_edge_theilsen_endpoints_tform[:, 1],
                    marker='+', s=ms, color="b")
        plt.scatter(glob_quadrant_D.v_edge_theilsen_endpoints_tform[:, 0],
                    glob_quadrant_D.v_edge_theilsen_endpoints_tform[:, 1],
                    marker='*', s=ms, color="g")
        plt.scatter(glob_quadrant_D.h_edge_theilsen_endpoints_tform[:, 0],
                    glob_quadrant_D.h_edge_theilsen_endpoints_tform[:, 1],
                    marker='+', s=ms, color="b")
        plt.show()

    return
