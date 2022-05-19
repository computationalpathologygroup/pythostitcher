import numpy as np
from .rotate_cp import rotate_cp
from .get_cost_functions import get_cost_functions
import tqdm

def compute_global_cost(imgCenters, tformedEdges_rc_world, tformedLines_xy_world, histograms,
                        intensities, adjustmentTform, ishpc, displayStateAndCost, simpleHistSize,
                        costFcnToUse, overhangPenalty, overlapAndUnderlapWeight, displayOpts,
                        samplingDelta, outerPointWeight):

    debug = False
    intensityCostWeight = 1 - overlapAndUnderlapWeight

    centerA_xy = imgCenters[0]
    centerB_xy = imgCenters[1]
    centerC_xy = imgCenters[2]
    centerD_xy = imgCenters[3]
    CoM = np.mean(imgCenters)

    # Unpack edges
    edge1A_rc_world = tformedEdges_rc_world["edge1A_rc"]
    edge2A_rc_world = tformedEdges_rc_world["edge1B_rc"]
    edge1B_rc_world = tformedEdges_rc_world["edge1C_rc"]
    edge2B_rc_world = tformedEdges_rc_world["edge1D_rc"]
    edge1C_rc_world = tformedEdges_rc_world["edge2A_rc"]
    edge2C_rc_world = tformedEdges_rc_world["edge2B_rc"]
    edge1D_rc_world = tformedEdges_rc_world["edge2C_rc"]
    edge2D_rc_world = tformedEdges_rc_world["edge2D_rc"]

    # Unpack lines
    line1A_xy_world = tformedLines_xy_world["line1A_xy"]
    line2A_xy_world = tformedLines_xy_world["line1B_xy"]
    line1B_xy_world = tformedLines_xy_world["line1C_xy"]
    line2B_xy_world = tformedLines_xy_world["line1D_xy"]
    line1C_xy_world = tformedLines_xy_world["line2A_xy"]
    line2C_xy_world = tformedLines_xy_world["line2B_xy"]
    line1D_xy_world = tformedLines_xy_world["line2C_xy"]
    line2D_xy_world = tformedLines_xy_world["line2D_xy"]

    # Rotate and apply translation to edges and lines
    edge1A_rc_tformed = edge1A_rc_world
    edge2A_rc_tformed = edge2A_rc_world
    line1A_xy_tformed = line1A_xy_world
    line2A_xy_tformed = line2A_xy_world

    # Rotate edges
    edge1B_xy_tformed = rotate_cp(centerB_xy, edge1B_rc_world[:, [1, 0]], adjustmentTform[2])
    edge1B_xy_tformed = edge1B_xy_tformed + np.tile([adjustmentTform[0], adjustmentTform[1]], [np.shape(edge1B_xy_tformed)[0], 1])
    edge1B_rc_tformed = edge1B_xy_tformed[:, [1, 0]]
    edge2B_xy_tformed = rotate_cp(centerB_xy, edge2B_rc_world[:, [1, 0]], adjustmentTform[2])
    edge2B_xy_tformed = edge2B_xy_tformed + np.tile([adjustmentTform[0], adjustmentTform[1]], [np.shape(edge2B_xy_tformed)[0], 1])
    edge2B_rc_tformed = edge2B_xy_tformed[:, [1, 0]]

    edge1C_xy_tformed = rotate_cp(centerC_xy, edge1C_rc_world[:, [1, 0]], adjustmentTform[5])
    edge1C_xy_tformed = edge1C_xy_tformed + np.tile([adjustmentTform[3], adjustmentTform[4]], [np.shape(edge1C_xy_tformed)[0], 1])
    edge1C_rc_tformed = edge1C_xy_tformed[:, [1, 0]]
    edge2C_xy_tformed = rotate_cp(centerC_xy, edge2C_rc_world[:, [1, 0]], adjustmentTform[5])
    edge2C_xy_tformed = edge2C_xy_tformed + np.tile([adjustmentTform[3], adjustmentTform[4]], [np.shape(edge2C_xy_tformed)[0], 1])
    edge2C_rc_tformed = edge2C_xy_tformed[:, [1, 0]]

    edge1D_xy_tformed = rotate_cp(centerD_xy, edge1D_rc_world[:, [1, 0]], adjustmentTform[8])
    edge1D_xy_tformed = edge1D_xy_tformed + np.tile([adjustmentTform[6], adjustmentTform[7]], [np.shape(edge1D_xy_tformed)[0], 1])
    edge1D_rc_tformed = edge1D_xy_tformed[:, [1, 0]]
    edge2D_xy_tformed = rotate_cp(centerD_xy, edge2D_rc_world[:, [1, 0]], adjustmentTform[8])
    edge2D_xy_tformed = edge2D_xy_tformed + np.tile([adjustmentTform[6], adjustmentTform[7]], [np.shape(edge2D_xy_tformed)[0], 1])
    edge2D_rc_tformed = edge2D_xy_tformed[:, [1, 0]]

    # Rotate lines
    line1B_xy_tformed = rotate_cp(centerB_xy, line1B_xy_world, adjustmentTform[2])
    line1B_xy_tformed = line1B_xy_tformed + np.tile([adjustmentTform[0], adjustmentTform[1]], [np.shape(line1B_xy_tformed)[0], 1])
    line2B_xy_tformed = rotate_cp(centerB_xy, line2B_xy_world, adjustmentTform[2])
    line2B_xy_tformed = line2B_xy_tformed + np.tile([adjustmentTform[0], adjustmentTform[1]], [np.shape(line2B_xy_tformed)[0], 1])

    line1C_xy_tformed = rotate_cp(centerC_xy, line1C_xy_world, adjustmentTform[5])
    line1C_xy_tformed = line1C_xy_tformed + np.tile([adjustmentTform[3], adjustmentTform[4]], [np.shape(line1C_xy_tformed)[0], 1])
    line2C_xy_tformed = rotate_cp(centerC_xy, line2C_xy_world, adjustmentTform[5])
    line2C_xy_tformed = line2C_xy_tformed + np.tile([adjustmentTform[3], adjustmentTform[4]], [np.shape(line2C_xy_tformed)[0], 1])

    line1D_xy_tformed = rotate_cp(centerD_xy, line1D_xy_world, adjustmentTform[8])
    line1D_xy_tformed = line1D_xy_tformed + np.tile([adjustmentTform[6], adjustmentTform[7]], [np.shape(line1D_xy_tformed)[0], 1])
    line2D_xy_tformed = rotate_cp(centerD_xy, line2D_xy_world, adjustmentTform[8])
    line2D_xy_tformed = line2D_xy_tformed + np.tile([adjustmentTform[6], adjustmentTform[7]], [np.shape(line2D_xy_tformed)[0], 1])

    # Packup edges in list
    edges_rc_tformed = [[edge1A_rc_tformed, edge1C_rc_tformed],
                        [edge1B_rc_tformed, edge1D_rc_tformed],
                        [edge1C_rc_tformed, edge1A_rc_tformed],
                        [edge1D_rc_tformed, edge1B_rc_tformed],
                        [edge2A_rc_tformed, edge2B_rc_tformed],
                        [edge2B_rc_tformed, edge2A_rc_tformed],
                        [edge2C_rc_tformed, edge2D_rc_tformed],
                        [edge2D_rc_tformed, edge2C_rc_tformed]]

    lines_xy_tformed = [[line1A_xy_tformed, line1C_xy_tformed],
                        [line1B_xy_tformed, line1D_xy_tformed],
                        [line1C_xy_tformed, line1A_xy_tformed],
                        [line1D_xy_tformed, line1B_xy_tformed],
                        [line2A_xy_tformed, line2B_xy_tformed],
                        [line2B_xy_tformed, line2A_xy_tformed],
                        [line2C_xy_tformed, line2D_xy_tformed],
                        [line2D_xy_tformed, line2C_xy_tformed]]

    whichQuadrants = ['topBottom', 'topBottom', 'bottomTop', 'bottomTop',
                      'leftRight', 'rightLeft', 'leftRight', 'rightLeft']

    # Preallocate some dicts
    intensityCosts = dict()
    edge_sampleIdxs = dict()
    overlapAndUnderlapCosts = dict()
    overhangIdxs = dict()

    if costFcnToUse == "simple_hists":
        hists1A = histograms["hist_1A"]
        hists1B = histograms["hist_1B"]
        hists1C = histograms["hist_1C"]
        hists1D = histograms["hist_1D"]

        hists2A = histograms["hist_2A"]
        hists2B = histograms["hist_2B"]
        hists2C = histograms["hist_2C"]
        hists2D = histograms["hist_2D"]

        hists_cell = [[hists1A, hists1C],
                      [hists1B, hists1D],
                      [hists1C, hists1A],
                      [hists1D, hists1B],
                      [hists2A, hists2B],
                      [hists2B, hists2A],
                      [hists2C, hists2D],
                      [hists2D, hists2C]]

        for i in tqdm.tqdm(range(8)):
            intensityCosts[str(i)], edge_sampleIdxs[str(i)], overlapAndUnderlapCosts[str(i)], overhangIdxs[str(i)] = get_cost_functions(
                edges_rc_tformed[i][0], edges_rc_tformed[i][1], lines_xy_tformed[i][0], lines_xy_tformed[i][1],
                whichQuadrants[i], hists_cell[i][0], hists_cell[i][1], samplingDelta, debug, CoM, outerPointWeight)

            # Not sure why this variable is needed
            # intensityCosts[i][overhangIdxs[i]] = overhangPenalty

    elif costFcnToUse == "raw_intensities":
        intensities1A = intensities["1A"]
        intensities1B = intensities["1B"]
        intensities1C = intensities["1C"]
        intensities1D = intensities["1D"]
        intensities2A = intensities["2A"]
        intensities2B = intensities["2B"]
        intensities2C = intensities["2C"]
        intensities2D = intensities["2D"]

        intensities_cell = [[intensities1A, intensities1C],
                            [intensities1B, intensities1D],
                            [intensities1C, intensities1A],
                            [intensities1D, intensities1B],
                            [intensities2A, intensities2B],
                            [intensities2B, intensities2A],
                            [intensities2C, intensities2D],
                            [intensities2D, intensities2C]]

        for i in tqdm.tqdm(range(8)):
            intensityCosts[str(i)], edge_sampleIdxs[str(i)], overlapAndUnderlapCosts[str(i)], overhangIdxs[str(i)] = get_cost_functions(
                edges_rc_tformed[i][0], edges_rc_tformed[i][1], lines_xy_tformed[i][0], lines_xy_tformed[i][1],
                whichQuadrants[i], intensities_cell[i][0], intensities_cell[i][1], samplingDelta, debug, CoM, outerPointWeight)

            # Not sure why this variable is needed
            # intensityCosts[i][overhangIdxs[i]] = overhangPenalty

    else:
        raise ValueError(f"Unexpected cost function, options are: [simpleHists, rawIntensities]")

    # Compute mean and overall costs
    meanOverlapAndUnderlapCosts = np.zeros((8))
    meanIntensityCosts = np.zeros((8))

    for j in range(8):
        meanOverlapAndUnderlapCosts[j] = np.mean([value for value in overlapAndUnderlapCosts.values()])
        meanIntensityCosts[j] = np.mean([value for value in intensityCosts.values()])

    totalOverlapAndUnderlapCost = np.sqrt(np.mean(meanOverlapAndUnderlapCosts))
    totalIntensityCost = np.mean(meanIntensityCosts)

    cost = intensityCostWeight * totalIntensityCost + overlapAndUnderlapWeight * totalOverlapAndUnderlapCost

    allCosts = dict()
    allCosts["name"] = ["1A", "1B", "1C", "1D", "2A", "2B", "2C", "2D"]
    allCosts["intensityCost"] = meanIntensityCosts
    allCosts["overlapAndUnderlapCost"] = meanOverlapAndUnderlapCosts

    return -totalOverlapAndUnderlapCost
    # return cost, totalIntensityCost, totalOverlapAndUnderlapCost, ACBD, ACBD_color_cropped, allCosts