import pygad
import numpy as np
import copy
import matplotlib.pyplot as plt
import itertools
import warnings

from shapely.geometry import Polygon

from .plot_tools import plot_sampled_patches
from .recombine_quadrants import recombine_quadrants
from .transformations import warp_2d_points, warp_image


def genetic_algorithm(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, initial_tform):
    """
    Function that runs a genetic algorithm using the pygad module. This function will use a global assembly method
    where the stitching for all quadrants is optimised at once.

    Input:
    - all quadrants
    - dict with parameters
    - initial transformation matrix

    Output:
    - dict with optimised transformation matrix and the fitness of that transformation
    """

    # Make some variables global for access by fitness function. The fitness function in pygad must be built to not
    # accept any other input parameters than the solution and solution index, but we need the quadrants to compute
    # fitness.
    global global_quadrant_A, global_quadrant_B, global_quadrant_C, global_quadrant_D
    global global_parameters, num_gen, global_init_tform
    global_quadrant_A = copy.deepcopy(quadrant_A)
    global_quadrant_B = copy.deepcopy(quadrant_B)
    global_quadrant_C = copy.deepcopy(quadrant_C)
    global_quadrant_D = copy.deepcopy(quadrant_D)
    global_parameters = copy.deepcopy(parameters)
    global_init_tform = copy.deepcopy(initial_tform)

    #
    if parameters["iteration"] == 0 and 'distance_scaling' not in globals():
        global distance_scaling
        distance_scaling = dict()
        distance_scaling["distance_scaling_hor_required"] = True
        distance_scaling["distance_scaling_ver_required"] = True

    # Initialize parameters
    tform_combi = [*initial_tform[quadrant_A.quadrant_name][:-2],
                   *initial_tform[quadrant_B.quadrant_name][:-2],
                   *initial_tform[quadrant_C.quadrant_name][:-2],
                   *initial_tform[quadrant_D.quadrant_name][:-2]]
    num_genes = len(tform_combi)
    ga_tform = np.zeros(num_genes)
    init_fitness = fitness_func(solution=ga_tform, solution_idx=0)
    num_sol = parameters["n_solutions"]
    num_gen = parameters["n_generations"]
    keep_parents = parameters["n_parents"]
    parents_mating = parameters["n_mating"]
    parents = parameters["parent_selection"]
    p_crossover = parameters["p_crossover"]
    crossover_type = parameters["crossover_type"]
    p_mutation = parameters["p_mutation"]
    mutation_type = parameters["mutation_type"]

    # Cap solution ranges to plausible values. The param_range variable denotes the range for the mutation values
    # that can be added/substracted from the genes in each parent.
    t_range = parameters["translation_range"][parameters["iteration"]] * int(np.mean(quadrant_A.tform_image.shape))
    a_range = parameters["angle_range"][parameters["iteration"]]
    angles = [False, False, True]*4
    lb = [int(x - a_range) if is_angle else int(x - t_range) for x, is_angle in zip(ga_tform, angles)]
    ub = [int(x + a_range) if is_angle else int(x + t_range) for x, is_angle in zip(ga_tform, angles)]
    param_range = [np.arange(l, u, step=0.1) if a else np.arange(l, u, step=1) for l, u, a in zip(lb, ub, angles)]

    # Perform random initialization of the first generation. The range for these values has been established
    # empirically. The seed is used to ensure that we start with the same initial population for every testrun.
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
        num_generations=num_gen,                    # num generations to optimize
        fitness_func=fitness_func,                  # optimization function
        initial_population=init_pop,                # gene values for first population
        gene_space=param_range,                     # parameter search range
        keep_parents=keep_parents,                  # num of parents that proceed to next generation unaltered
        num_parents_mating=parents_mating,          # num parents that produce offspring
        parent_selection_type=parents,              # function to select parents for offspring
        crossover_type=crossover_type,              # (M) method how genes between parents are combined
        crossover_probability=p_crossover,          # probability that a parent is chosen for crossover
        mutation_type=mutation_type,                # method how gene values are mutated
        mutation_probability=p_mutation,            # probability that a gene mutates
        on_generation=plot_best_sol_per_gen,        # plot the result after every N generations
        save_best_solutions=True,                   # must be True in order to use the callback above
        suppress_warnings=True,                     # suppress annoying and irrelevant warnings
        stop_criteria=None                          # this could later be set to stop after loss plateaus
    )

    # Run genetic algorithm
    ga_instance.run()

    # Plot fitness and retrieve best solution
    ga_instance.plot_fitness()
    solution, solution_fitness, _ = ga_instance.best_solution()

    # Get genetic algorithm solution per quadrant
    sol_A = [np.round(s, 1) for s in solution[:3]]
    sol_B = [np.round(s, 1) for s in solution[3:6]]
    sol_C = [np.round(s, 1) for s in solution[6:9]]
    sol_D = [np.round(s, 1) for s in solution[9:]]

    # Combine genetic algorithm solution with initial transformation
    combo_tform_a = [
        *[a + b for a, b in zip(global_init_tform[global_quadrant_A.quadrant_name][:3], sol_A)],
        *global_init_tform[global_quadrant_A.quadrant_name][3:]
    ]
    combo_tform_b = [
        *[a + b for a, b in zip(global_init_tform[global_quadrant_B.quadrant_name][:3], sol_B)],
        *global_init_tform[global_quadrant_B.quadrant_name][3:]
    ]
    combo_tform_c = [
        *[a + b for a, b in zip(global_init_tform[global_quadrant_C.quadrant_name][:3], sol_C)],
        *global_init_tform[global_quadrant_C.quadrant_name][3:]
    ]
    combo_tform_d = [
        *[a + b for a, b in zip(global_init_tform[global_quadrant_D.quadrant_name][:3], sol_D)],
        *global_init_tform[global_quadrant_D.quadrant_name][3:]
    ]

    # Save some results for later plotting
    if parameters["iteration"] == 0:
        parameters["GA_fitness"].append(init_fitness)
    parameters["GA_fitness"].append(solution_fitness)

    # Save final transformation for each quadrant
    ga_dict = dict()
    ga_dict["fitness"] = solution_fitness
    tforms = [combo_tform_a, combo_tform_b, combo_tform_c, combo_tform_d]
    for t, key in zip(tforms, parameters["filenames"].keys()):
        ga_dict[key] = t

    return ga_dict


def fitness_func(solution, solution_idx):
    """
    Custom function to compute the fitness related to the genetic algorithm. The genetic algorithm will provide
    a solution for the transformation matrix which can be used to compute the cost. A better transformation
    matrix should be associated with a lower cost and thus a higher fitness.

    Although histogram matching and overlap are currently supported as cost functions, these require warping of much
    bigger sets of coordinates or even images and are therefore much more computationally expensive. For now it is
    recommended to only use the distance cost function.

    Input:
        - solution (list of length num_genes)
        - solution_idx (mandatory input but is not used)

    Output:
        - fitness (note that the genetic algorithm in pygad aims to maximize the fitness function and thus this
          function must return the fitness instead of the cost)
    """

    global global_quadrant_A, global_quadrant_B, global_quadrant_C, global_quadrant_D, global_parameters
    global distance_scaling

    # General cost function global_parameters
    global_parameters["center_of_mass"] = np.mean(global_parameters["image_centers"], axis=0)

    # Get solution per quadrant
    sol_A = [*solution[:3], global_quadrant_A.image_center_peri, global_init_tform[global_quadrant_A.quadrant_name][4]]
    sol_B = [*solution[3:6], global_quadrant_B.image_center_peri, global_init_tform[global_quadrant_B.quadrant_name][4]]
    sol_C = [*solution[6:9], global_quadrant_C.image_center_peri, global_init_tform[global_quadrant_C.quadrant_name][4]]
    sol_D = [*solution[9:], global_quadrant_D.image_center_peri, global_init_tform[global_quadrant_D.quadrant_name][4]]

    # Combine genetic algorithm solution with initial transformation
    combo_tform_a = [
        *[a + b for a, b in zip(global_init_tform[global_quadrant_A.quadrant_name][:3], sol_A)],
        *global_init_tform[global_quadrant_A.quadrant_name][3:]
    ]
    combo_tform_b = [
        *[a + b for a, b in zip(global_init_tform[global_quadrant_B.quadrant_name][:3], sol_B)],
        *global_init_tform[global_quadrant_B.quadrant_name][3:]
    ]
    combo_tform_c = [
        *[a + b for a, b in zip(global_init_tform[global_quadrant_C.quadrant_name][:3], sol_C)],
        *global_init_tform[global_quadrant_C.quadrant_name][3:]
    ]
    combo_tform_d = [
        *[a + b for a, b in zip(global_init_tform[global_quadrant_D.quadrant_name][:3], sol_D)],
        *global_init_tform[global_quadrant_D.quadrant_name][3:]
    ]

    # Apply tform to several attributes of the quadrant
    global_quadrant_A = apply_new_transform(
        quadrant=global_quadrant_A,
        sol_tform=sol_A,
        combo_tform=combo_tform_a,
        tform_image=False
    )
    global_quadrant_B = apply_new_transform(
        quadrant=global_quadrant_B,
        sol_tform=sol_B,
        combo_tform=combo_tform_b,
        tform_image=False
    )
    global_quadrant_C = apply_new_transform(
        quadrant=global_quadrant_C,
        sol_tform=sol_C,
        combo_tform=combo_tform_c,
        tform_image=False
    )
    global_quadrant_D = apply_new_transform(
        quadrant=global_quadrant_D,
        sol_tform=sol_D,
        combo_tform=combo_tform_d,
        tform_image=False
    )

    """
    # Cost function that penalizes dissimilar histograms along the edge of neighbouring quadrants
    histogram_cost = hist_cost_function(
        quadrant_A=global_quadrant_A,
        quadrant_B=global_quadrant_B,
        quadrant_C=global_quadrant_C,
        quadrant_D=global_quadrant_D,
        parameters=global_parameters,
        plot=False
    )
    
    
    # Cost function that penalizes a high degree of overlap between quadrants
    overlap_cost = overlap_cost_function(
        quadrant_A=global_quadrant_A,
        quadrant_B=global_quadrant_B,
        quadrant_C=global_quadrant_C,
        quadrant_D=global_quadrant_D,
    )
    """
    # Cost function that penalizes a large distance between endpoints of quadrants
    distance_cost = distance_cost_function(
        quadrant_A=global_quadrant_A,
        quadrant_B=global_quadrant_B,
        quadrant_C=global_quadrant_C,
        quadrant_D=global_quadrant_D,
    )

    total_cost = distance_cost

    return 1/total_cost


def apply_new_transform(quadrant, sol_tform, combo_tform, tform_image=False):
    """
    Custom function to apply the newly acquired genetic algorithm solution to several attributes of the quadrant. These
    attributes will subsequently be used for computing the cost

    Input:
    - Quadrants
    - Transformation matrix

    Output:
    - Transformed quadrants and lines
    """

    # Ensure correct formats
    center_sol = tuple(np.squeeze(sol_tform[3]))
    center_combo = tuple(np.squeeze(combo_tform[3]))

    # Apply tform to theilsen endpoints
    quadrant.h_edge_theilsen_endpoints_tform = warp_2d_points(
        src=quadrant.h_edge_theilsen_endpoints,
        center=center_sol,
        rotation=sol_tform[2],
        translation=sol_tform[:2]
    )

    quadrant.v_edge_theilsen_endpoints_tform = warp_2d_points(
        src=quadrant.v_edge_theilsen_endpoints,
        center=center_sol,
        rotation=sol_tform[2],
        translation=sol_tform[:2]
    )

    # Apply tform to mask coordinates. Append first element to coordinate list to ensure closed polygon.
    quadrant.mask_contour_tform = warp_2d_points(
        src=quadrant.mask_contour,
        center=center_combo,
        rotation=combo_tform[2],
        translation=combo_tform[:2]
    )
    quadrant.mask_contour_tform = list(quadrant.mask_contour_tform) + list([quadrant.mask_contour_tform[0]])

    # Apply tform to image center
    quadrant.image_center_post = warp_2d_points(
        src=quadrant.image_center_pre,
        center=center_combo,
        rotation=combo_tform[2],
        translation=combo_tform[:2]
    )

    # Apply tform to image when required
    if tform_image:
        quadrant.colour_image = warp_image(
            src=quadrant.colour_image_original,
            center=center_combo,
            rotation=combo_tform[2],
            translation=combo_tform[:2],
            output_shape=combo_tform[4]
        )

    return quadrant


def hist_cost_function(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters, plot=False):
    """
    Custom function which penalizes dissimilarity between histograms of patches alongside a stitching edge. This
    function loops over all edges and will extract a patch from the upper and lower side from horizontal lines and
    from the left and right side from vertical lines. When quadrants are aligned well, it is expected that the
    histograms of these patches should be approximately similar.

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
    images = [quadrant_A.colour_image, quadrant_B.colour_image, quadrant_C.colour_image, quadrant_D.colour_image]
    total_im = recombine_quadrants(images=images)

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
        prob_dens_upper = [h[0] / np.sum(h[0]) if np.sum(h[0]) != 0 else np.zeros(nbins) for h in histograms_upper]
        prob_dens_lower = [h[0] / np.sum(h[0]) if np.sum(h[0]) != 0 else np.zeros(nbins) for h in histograms_lower]

        # Compute difference between probability density function. For probability density functions of an
        # empty patch we set the cost to the maximum value of 2.
        summed_diff = [2 if (np.sum(prob1) == 0 and np.sum(prob2) == 0) else np.sum(np.abs(prob1 - prob2))
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
        prob_dens_left = [h[0] / np.sum(h[0]) if np.sum(h[0]) != 0 else np.zeros(nbins) for h in histograms_left]
        prob_dens_right = [h[0] / np.sum(h[0]) if np.sum(h[0]) != 0 else np.zeros(nbins) for h in histograms_right]

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


def overlap_cost_function(quadrant_A, quadrant_B, quadrant_C, quadrant_D):
    """
    Custom function to compute the overlap between the quadrants. This is implemented using polygons rather than
    the transformed images as this is an order of magnitude faster.

    Note that the current implementation provides an approximation of the overlap rather than the exact amount as
    overlap is only calculated for quadrant pairs and not quadrant triplets (i.e. if there is overlap between
    quadrant ACD this can be counted multiple times due to inclusion in the AC, AD and CD pairs).

    Input:
        - All four quadrants

    Output:
        - Cost
    """

    # Set some initial parameters
    keys = ["A", "B", "C", "D"]
    combinations = itertools.combinations(keys, 2)
    quadrants = [quadrant_A, quadrant_B, quadrant_C, quadrant_D]
    poly_dict = dict()
    total_area = 0
    total_overlap = 0
    weighting_factor = 10

    # Create a polygon from the transformed mask contour and compute its area
    for quadrant, key in zip(quadrants, keys):
        poly_dict[key] = Polygon(quadrant.mask_contour_tform)
        total_area += poly_dict[key].area

    # Return a cost of 0 when one of the polygons is invalid and thus overlap cost computation is not possible
    if not all([p.is_valid for p in poly_dict.values()]):
        return 0

    # Compute overlap between all possible quadrant pairs
    for combo in combinations:
        overlap_polygon = poly_dict[combo[0]].intersection(poly_dict[combo[1]])
        total_overlap += overlap_polygon.area

    # Compute relative overlap and apply weighting factor
    relative_overlap = total_overlap/total_area
    cost = relative_overlap * weighting_factor

    # Santiy check that the overlap doesn't exceed 100%
    assert relative_overlap < 1, "Overlap cannot be greater than 100%"

    return cost


def distance_cost_function(quadrant_A, quadrant_B, quadrant_C, quadrant_D):
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

    # Create variable for scaling the cost function with respect to the resolution
    global global_parameters
    global distance_scaling

    # Define pairs to loop over
    hline_pairs = [[quadrant_A, quadrant_C], [quadrant_B, quadrant_D]]
    vline_pairs = [[quadrant_A, quadrant_B], [quadrant_C, quadrant_D]]

    # Distance scaling parameters
    distance_costs = []
    res_scaling = global_parameters["resolution_scaling"][global_parameters["iteration"]]
    hline_keys_inner = ["AC_inner", "BD_inner"]
    hline_keys_outer = ["AC_outer", "BD_outer"]
    vline_keys_inner = ["AB_inner", "CD_inner"]
    vline_keys_outer = ["AB_outer", "CD_outer"]

    # Loop over horizontal lines
    for quadrants, hline_key_inner, hline_key_outer in zip(hline_pairs, hline_keys_inner, hline_keys_outer):

        # Extract quadrants
        quadrant1 = quadrants[0]
        quadrant2 = quadrants[1]

        # Get the lines from the quadrant
        hline1_pts = quadrant1.h_edge_theilsen_endpoints_tform
        hline2_pts = quadrant2.h_edge_theilsen_endpoints_tform

        # Calculate the distance of all points to the center of mass to find out which points are located at the center
        # side of the line.
        line1_distsFromCoM = [np.linalg.norm(hline1_pts[0] - global_parameters["center_of_mass"]),
                              np.linalg.norm(hline1_pts[1] - global_parameters["center_of_mass"])]
        line2_distsFromCoM = [np.linalg.norm(hline2_pts[0] - global_parameters["center_of_mass"]),
                              np.linalg.norm(hline2_pts[1] - global_parameters["center_of_mass"])]

        # Get indices of inner and outer points of both lines
        innerPtIdx_line1 = np.argmin(line1_distsFromCoM)
        outerPtIdx_line1 = 1 if innerPtIdx_line1 == 0 else 0

        innerPtIdx_line2 = np.argmin(line2_distsFromCoM)
        outerPtIdx_line2 = 1 if innerPtIdx_line2 == 0 else 0

        # Get the inner and outer points.
        line1_innerPt = hline1_pts[innerPtIdx_line1]
        line1_outerPt = hline1_pts[outerPtIdx_line1]
        line2_innerPt = hline2_pts[innerPtIdx_line2]
        line2_outerPt = hline2_pts[outerPtIdx_line2]

        # Obtain a scaling factor to normalize the distance cost across multiple resolutions. This scaling factor
        # consists of the initial cost at the lowest resolution which is then extrapolated to what this cost
        # would be on higher resolutions.
        if global_parameters["iteration"] == 0 and distance_scaling["distance_scaling_hor_required"]:
            distance_scaling[hline_key_inner] = np.round(np.linalg.norm(line1_innerPt - line2_innerPt), 2)
            distance_scaling[hline_key_outer] = np.round(np.linalg.norm(line1_outerPt - line2_outerPt), 2)

        # Handle cases where the first genetic algorithm optimalization has already been performed in a different
        # run and where the actual scaling factor is hence not available.
        elif global_parameters["iteration"] > 0 and distance_scaling["distance_scaling_hor_required"]:
            distance_scaling[hline_key_inner] = 1
            distance_scaling[hline_key_outer] = 1
            warnings.warn("Distance cost is not scaled properly")

        # Get resolution specific scaling factor
        scaling_inner = distance_scaling[hline_key_inner]*res_scaling
        scaling_outer = distance_scaling[hline_key_outer]*res_scaling

        # Compute edge_distance_costs as sum of distances
        inner_point_weight = 1 - global_parameters["outer_point_weight"]
        inner_point_norm = (np.linalg.norm(line1_innerPt - line2_innerPt) / scaling_inner) ** 2
        outer_point_norm = (np.linalg.norm(line1_outerPt - line2_outerPt) / scaling_outer) ** 2
        combined_costs = inner_point_weight * inner_point_norm + \
                         global_parameters["outer_point_weight"] * outer_point_norm
        distance_costs.append(combined_costs)

    # Loop over vertical lines
    for quadrants, vline_key_inner, vline_key_outer in zip(vline_pairs, vline_keys_inner, vline_keys_outer):

        # Extract quadrants
        quadrant1 = quadrants[0]
        quadrant2 = quadrants[1]

        # Get the lines from the quadrants
        vline1_pts = quadrant1.v_edge_theilsen_endpoints_tform
        vline2_pts = quadrant2.v_edge_theilsen_endpoints_tform

        # Calculate distance from center of mass
        line1_distsFromCoM = [np.linalg.norm(vline1_pts[0] - global_parameters["center_of_mass"]),
                              np.linalg.norm(vline1_pts[1] - global_parameters["center_of_mass"])]
        line2_distsFromCoM = [np.linalg.norm(vline2_pts[0] - global_parameters["center_of_mass"]),
                              np.linalg.norm(vline2_pts[1] - global_parameters["center_of_mass"])]

        # Get indices of inner and outer points of both lines
        innerPtIdx_line1 = np.argmin(line1_distsFromCoM)
        outerPtIdx_line1 = 1 if innerPtIdx_line1 == 0 else 0

        innerPtIdx_line2 = np.argmin(line2_distsFromCoM)
        outerPtIdx_line2 = 1 if innerPtIdx_line2 == 0 else 0

        # Get the inner and outer points. We divide this by the scaling to account for the increased distance due
        # to the higher resolutions.
        line1_innerPt = vline1_pts[innerPtIdx_line1]
        line1_outerPt = vline1_pts[outerPtIdx_line1]
        line2_innerPt = vline2_pts[innerPtIdx_line2]
        line2_outerPt = vline2_pts[outerPtIdx_line2]

        # Obtain a scaling factor to normalize the distance cost across multiple resolutions. This scaling factor
        # consists of the initial cost at the lowest resolution which is then extrapolated to what this cost
        # would be on higher resolutions.
        if global_parameters["iteration"] == 0 and distance_scaling["distance_scaling_ver_required"]:
            distance_scaling[vline_key_inner] = np.round(np.linalg.norm(line1_innerPt - line2_innerPt), 2)
            distance_scaling[vline_key_outer] = np.round(np.linalg.norm(line1_outerPt - line2_outerPt), 2)

        # Handle cases where the first genetic algorithm optimalization has already been performed in a different
        # run and where the actual scaling factor is hence not available.
        elif global_parameters["iteration"] > 0 and distance_scaling["distance_scaling_ver_required"]:
            distance_scaling[vline_key_inner] = 1
            distance_scaling[vline_key_outer] = 1
            warnings.warn("Distance cost is not scaled properly")

        # Get resolution specific scaling factor
        scaling_inner = distance_scaling[vline_key_inner]*res_scaling
        scaling_outer = distance_scaling[vline_key_outer]*res_scaling

        # Compute edge_distance_costs as sum of distances
        inner_point_weight = 1 - global_parameters["outer_point_weight"]
        inner_point_norm = (np.linalg.norm(line1_innerPt - line2_innerPt) / scaling_inner) ** 2
        outer_point_norm = (np.linalg.norm(line1_outerPt - line2_outerPt) / scaling_outer) ** 2
        combined_costs = inner_point_weight * inner_point_norm + global_parameters["outer_point_weight"] * outer_point_norm
        distance_costs.append(combined_costs)

    # Distance scaling should be computed now
    if len(distance_scaling.keys()) > 2:
        distance_scaling["distance_scaling_hor_required"] = False
        distance_scaling["distance_scaling_ver_required"] = False

    cost = np.mean(distance_costs)

    return cost

def plot_best_sol_per_gen(ga):
    """
    Custom function to show the best stitching result every N generations.

    Input:
        - Genetic algorithm instance

    Output:
        - Figure showing the stitching result
    """

    global global_quadrant_A, global_quadrant_B, global_quadrant_C, global_quadrant_D

    # Only plot the result every N generations
    gen = ga.generations_completed
    n = 50

    if gen % n == 0:

        # Get best solution
        solution, fitness, _ = ga.best_solution()

        # Get solution per quadrant
        sol_A = [*solution[:3], global_quadrant_A.image_center_peri, global_init_tform[global_quadrant_A.quadrant_name][4]]
        sol_B = [*solution[3:6], global_quadrant_B.image_center_peri, global_init_tform[global_quadrant_B.quadrant_name][4]]
        sol_C = [*solution[6:9], global_quadrant_C.image_center_peri, global_init_tform[global_quadrant_C.quadrant_name][4]]
        sol_D = [*solution[9:], global_quadrant_D.image_center_peri, global_init_tform[global_quadrant_D.quadrant_name][4]]

        # Combine genetic algorithm solution with initial transformation
        combo_tform_a = [
            *[a + b for a, b in zip(global_init_tform[global_quadrant_A.quadrant_name][:3], sol_A)],
            *global_init_tform[global_quadrant_A.quadrant_name][3:]
        ]
        combo_tform_b = [
            *[a + b for a, b in zip(global_init_tform[global_quadrant_B.quadrant_name][:3], sol_B)],
            *global_init_tform[global_quadrant_B.quadrant_name][3:]
        ]
        combo_tform_c = [
            *[a + b for a, b in zip(global_init_tform[global_quadrant_C.quadrant_name][:3], sol_C)],
            *global_init_tform[global_quadrant_C.quadrant_name][3:]
        ]
        combo_tform_d = [
            *[a + b for a, b in zip(global_init_tform[global_quadrant_D.quadrant_name][:3], sol_D)],
            *global_init_tform[global_quadrant_D.quadrant_name][3:]
        ]

        # Apply tform to several attributes of the quadrant
        global_quadrant_A = apply_new_transform(
            quadrant=global_quadrant_A,
            sol_tform=sol_A,
            combo_tform=combo_tform_a,
            tform_image=True
        )
        global_quadrant_B = apply_new_transform(
            quadrant=global_quadrant_B,
            sol_tform=sol_B,
            combo_tform=combo_tform_b,
            tform_image=True
        )
        global_quadrant_C = apply_new_transform(
            quadrant=global_quadrant_C,
            sol_tform=sol_C,
            combo_tform=combo_tform_c,
            tform_image=True
        )
        global_quadrant_D = apply_new_transform(
            quadrant=global_quadrant_D,
            sol_tform=sol_D,
            combo_tform=combo_tform_d,
            tform_image=True
        )

        # Get final image
        # EXPERIMENTAL
        images = [global_quadrant_A.colour_image,
                  global_quadrant_B.colour_image,
                  global_quadrant_C.colour_image,
                  global_quadrant_D.colour_image]
        total_im = recombine_quadrants(images=images)
        """
        total_im = recombine_quadrants(
            global_quadrant_A.colour_image,
            global_quadrant_B.colour_image,
            global_quadrant_C.colour_image,
            global_quadrant_D.colour_image
        )
        """

        # Plotting parameters
        ratio = global_parameters["resolutions"][global_parameters["iteration"]] / global_parameters["resolutions"][0]
        ms = np.sqrt(2500 * np.sqrt(ratio))

        # Make figure
        plt.figure()
        plt.title(f"Best result at generation {gen}/{num_gen}: fitness = {np.round(fitness, 2)}")
        plt.imshow(total_im, cmap="gray")
        plt.scatter(global_quadrant_A.v_edge_theilsen_endpoints_tform[:, 0],
                    global_quadrant_A.v_edge_theilsen_endpoints_tform[:, 1],
                    marker='*', s=ms, color="g")
        plt.scatter(global_quadrant_A.h_edge_theilsen_endpoints_tform[:, 0],
                    global_quadrant_A.h_edge_theilsen_endpoints_tform[:, 1],
                    marker='+', s=ms, color="b")
        plt.scatter(global_quadrant_B.v_edge_theilsen_endpoints_tform[:, 0],
                    global_quadrant_B.v_edge_theilsen_endpoints_tform[:, 1],
                    marker='*', s=ms, color="g")
        plt.scatter(global_quadrant_B.h_edge_theilsen_endpoints_tform[:, 0],
                    global_quadrant_B.h_edge_theilsen_endpoints_tform[:, 1],
                    marker='+', s=ms, color="b")
        plt.scatter(global_quadrant_C.v_edge_theilsen_endpoints_tform[:, 0],
                    global_quadrant_C.v_edge_theilsen_endpoints_tform[:, 1],
                    marker='*', s=ms, color="g")
        plt.scatter(global_quadrant_C.h_edge_theilsen_endpoints_tform[:, 0],
                    global_quadrant_C.h_edge_theilsen_endpoints_tform[:, 1],
                    marker='+', s=ms, color="b")
        plt.scatter(global_quadrant_D.v_edge_theilsen_endpoints_tform[:, 0],
                    global_quadrant_D.v_edge_theilsen_endpoints_tform[:, 1],
                    marker='*', s=ms, color="g")
        plt.scatter(global_quadrant_D.h_edge_theilsen_endpoints_tform[:, 0],
                    global_quadrant_D.h_edge_theilsen_endpoints_tform[:, 1],
                    marker='+', s=ms, color="b")
        plt.show()

    return