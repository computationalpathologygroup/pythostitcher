import numpy as np
import os
import matplotlib.pyplot as plt
# from tqdm import tqdm

from utils.preprocess import preprocess
from utils.map_high_res import map_high_res
from utils.optimize_stitch import optimize_stitch
from utils.quadrant_class import Quadrant


def setup_pythostitcher(parameters):

    """
    PythoStitcher is an automated and robust program for stitching tissue fragments
    into a whole histological section.
    
    Original paper: http://www.nature.com/articles/srep29906
    Original Matlab code by Greg Penzias, 2016
    Python implementation by Daan Schouten, 2022
    
    Input arguments
        * parameters dictionary with
            - filepaths to quadrants
            - patient name
            - slice name
            - directories for data/results
        
    Output
        * Final reconstructed histological image
        
    """

    # General input parameters
    parameters["debug"] = False
    parameters["ishpc"] = False
    parameters["display_opts"] = "EdgesAndLines"
    parameters["rev_name"] = "r1"
    # parameters["resolutions"] = [0.01, 0.05, 0.25, 1.0]
    parameters["resolutions"] = [0.05, 0.10, 0.25, 1.0]
    parameters["padsizes"] = [int(200*r) for r in parameters["resolutions"]]

    # Parameters related to the cost function
    parameters["overunderlap_weights"] = [0.01, 0.01, 0.011, 0.01]
    parameters["cost_range"] = [0, 2]
    parameters["overhang_penalty"] = parameters["cost_range"][-1]
    # parameters["cost_functions"] = ["raw_intensities", "raw_intensities", "simple_hists", "simple_hists"]
    parameters["cost_functions"] = ["simple_hists", "simple_hists", "simple_hists", "simple_hists"]
    parameters["nbins"] = 16
    # parameters["hist_sizes"] = [[1, 1], [1, 1], [21, 11], [81, 41]]
    parameters["hist_sizes"] = [[5, 3], [11, 5], [21, 11], [81, 41]]
    parameters["fraction_edge_length"] = 0.5
    parameters["outer_point_weight"] = 0.6

    # Optimization parameters
    parameters["translation_ranges"] = [1000*r for r in parameters["resolutions"]]
    #parameters["sampling_deltas"] = [1, 1, 5, 10]
    parameters["sampling_deltas"] = [1, 2, 5, 10]
    parameters["angle_range"] = 15

    print("\nPreprocessing data on multiple resolutions...")

    # Start with preprocessing data
    for iter in range(len(parameters["resolutions"])):

        # Set current iteration
        parameters["iteration"] = iter

        # Initiate all quadrants
        quadrant_A = Quadrant(quadrant_name = "UL",
                              kwargs=parameters)

        quadrant_B = Quadrant(quadrant_name="UR",
                              kwargs=parameters)

        quadrant_C = Quadrant(quadrant_name="LL",
                              kwargs=parameters)

        quadrant_D = Quadrant(quadrant_name="LR",
                              kwargs=parameters)

        # Preprocess all images
        preprocess(quadrant_A, quadrant_B, quadrant_C, quadrant_D)

    print("Finished!")

    # Optimize stitch for multiple resolutions
    for iter in range(len(parameters["resolutions"])):

        parameters["iteration"] = iter
        print(f"\nOptimizing stitch at resolution {parameters['resolutions'][iter]}")
        solution_fitness = optimize_stitch(parameters, plot=True)

    res_num = 3
    imA_t, imB_t, imC_t, imD_t, tform_objects, tform_edges = map_high_res(
        data_dir, results_dir, cases, res_num, versions,
        rev_name, padsizes, resolutions[res_num], "AutoStitch")

    return
