from utils.preprocess import preprocess
from utils.map_high_res import map_high_res
from utils.optimize_stitch import optimize_stitch
from utils.quadrant_class import Quadrant


def run_pythostitcher(parameters):
    """
    PythoStitcher is an automated and robust program for stitching prostate tissue fragments
    into a whole histological section.
    
    Original paper: http://www.nature.com/articles/srep29906
    Original Matlab code by Greg Penzias, 2016
    Python implementation by Daan Schouten, 2022
    
    Input arguments
    * dictionary with parameters
        
    Output
    * Final reconstructed histological image

    """

    # Start with preprocessing data
    print("\nPreprocessing data on multiple resolutions...")

    for i in range(len(parameters["resolutions"])):

        # Set current iteration
        parameters["iteration"] = i

        # Initiate all quadrants
        quadrant_A = Quadrant(quadrant_name="UL", kwargs=parameters)
        quadrant_B = Quadrant(quadrant_name="UR", kwargs=parameters)
        quadrant_C = Quadrant(quadrant_name="LL", kwargs=parameters)
        quadrant_D = Quadrant(quadrant_name="LR", kwargs=parameters)

        # Preprocess all images
        preprocess(quadrant_A, quadrant_B, quadrant_C, quadrant_D)

    print("Finished!")

    # Optimize stitch for multiple resolutions
    for i in range(len(parameters["resolutions"])):

        parameters["iteration"] = i
        print(f"\nOptimizing stitch at resolution {parameters['resolutions'][i]}")
        optimize_stitch(parameters, plot=True)

    ##### Last code to be implemented
    imA_t, imB_t, imC_t, imD_t, tform_objects, tform_edges = map_high_res(
        data_dir, results_dir, cases, res_num, versions,
        rev_name, padsizes, resolutions[res_num], "AutoStitch")

    return
