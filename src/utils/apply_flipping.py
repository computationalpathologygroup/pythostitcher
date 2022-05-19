import os
import numpy as np
from .get_resname import get_resname
from .get_filename import get_filename


def apply_flipping(results_dir, cases, caseIdx, sliceIdx, resNum, versions, revName):

    # Load in processed images
    caseName = cases["name"]
    sliceName = cases["slice_name"]

    res = versions["resolution_level"][resNum]
    # res = version.resolutionLevel
    resName = get_resname(res)

    dirpath_determineFlipping1 = f"{results_dir}/{caseName}/{sliceName}/determineFlipping"
    dirpath_determineFlipping2 = f"{results_dir}/{caseName}/{sliceName}/determineFlipping/{resName}"

    if not os.path.isdir(dirpath_determineFlipping1):
        os.mkdir(dirpath_determineFlipping1)

    # Make new directory if necessary
    if not os.path.isdir(dirpath_determineFlipping2):
        os.mkdir(dirpath_determineFlipping2)

    flipState = cases["toflip"]

    #### THIS CODE APPLIES FLIPPING BUT CAN BE MUCH MORE EFFICIENT.            ####
    #### EXCLUDE FOR NOW AND LATER PERFORM A REVISION WHEN FLIPPING IS NEEDED. ####

    ## Apply flipping to the preprocessed images
    # for i in range(len(versions["resolution_level"])):
    #     resName_ApplyFlipping = get_resname(versions["resolution_level"][i])
    #
    #     # Load preprocessing variables
    #     dirpath_preprocessed = f"{results_dir}/{caseName}/{sliceName}/{resName_ApplyFlipping}/preprocessed/"
    #     fname_preprocessed = get_filename(dirpath_preprocessed, versions["resolution_level"][i], revName)
    #
    #     preprocessedFilepath = f"{fname_preprocessed}.npy"
    #     flip_dict = np.load(preprocessedFilepath, allow_pickle=True).item()
    #
    #     print(flip_dict)
    #
    #     #### CODE BLOCK BELOW CANNOT EXECUTE DUE TO LOADING OF REQUIRED VARIABLES ####
    #     # Apply flipping.
    #     imFlips_orig = [flipA, flipB, flipC, flipD]
    #
    #     # The flipping that needs to be applied to achieve flipstate
    #     imFlips = [a!=b for a, b in zip(imFlips, flipState)]
    #
    #     flipA = imFlips[0]
    #     flipB = imFlips[1]
    #     flipC = imFlips[2]
    #     flipD = imFlips[3]
    #
    #     if 'flipA' in locals():
    #         imgA_g = np.flipud(imgA_g)
    #     if 'flipB' in locals():
    #         imgB_g = np.flipud(imgB_g)
    #     if 'flipC' in locals():
    #         imgC_g = np.flipud(imgC_g)
    #     if 'flipD' in locals():
    #         imgD_g = np.flipud(imgD_g)
    #
    #     if 'imgA' in locals():
    #         if 'flipA' in locals():
    #             imgA = np.flipud(imgA)
    #         if 'fliBA' in locals():
    #             imgB = np.flipud(imgB)
    #         if 'flipC' in locals():
    #             imgC = np.flipud(imgC)
    #         if 'flipD' in locals():
    #             imgD = np.flipud(imgD)

    # Load dictionary with images
    resName_ApplyFlipping = get_resname(versions["resolution_level"][i])
    dirpath_preprocessed = f"{results_dir}/{caseName}/{sliceName}/{resName_ApplyFlipping}/preprocessed/"
    fname_preprocessed = get_filename(dirpath_preprocessed, versions["resolution_level"][i], revName)

    preprocessedFilepath = f"{fname_preprocessed}.npy"
    flip_dict = np.load(preprocessedFilepath, allow_pickle=True).item()

    # Create a dict for saving the variables
    dictA = {}
    dictA_keys = ["imgA_g", "imgB_g", "imgC_g", "imgD_g", "quadrantNameA", "quadrantNameB", "quadrantNameC",
                  "quadrantNameD", "imgA", "imgB", "imgC", "imgD"]
    dict_keys_flip = ["flipA", "flipB", "flipC", "flipD"]

    for key in dictA_keys:
        dictA[key] = flip_dict[key]

    for key_flip, flip in zip(dict_keys_flip, flipState):
        dictA[key_flip] = flip

    dictB = {}
    dictA_keys = ["imgA_g", "imgB_g", "imgC_g", "imgD_g", "quadrantNameA", "quadrantNameB", "quadrantNameC",
                  "quadrantNameD"]

    for key in dictB_keys:
        dictB[key] = flip_dict[key]

    for key_flip, flip in zip(dict_keys_flip, flipState):
        dictB[key_flip] = flip

    # Overwrite preprocessing variables
    if 'imgA' in locals():
        np.save(preprocessedFilepath, dictA)
    else:
        np.save(preprocessedFilepath, dictB)

    return dictA, dictB