import numpy as np


def get_cost_functions(edge_tform_a, edge_tform_b, edge_tform_theilsen_a, edge_tform_theilsen_b,
                       histograms_a, histograms_b, quadrant, parameters, direction):

    # Make sure coordinates are integers and remove first and last edge point
    edge1_rc_world = np.round(edge_tform_a).astype("int")
    edge2_rc_world = np.round(edge_tform_b).astype("int")

    # Check whether lines are defined in same direction. This is required for histogram matching along the edge
    if direction == "horizontal":

        if (edge1_rc_world[-1][0] - edge1_rc_world[0][0]) < 0:
            edge1_left2right = False
        else:
            edge1_left2right = True
        if (edge2_rc_world[-1][0] - edge2_rc_world[0][0]) < 0:
            edge2_left2right =  False
        else:
            edge2_left2right = True

        # If the lines do not have the same direction, reverse one of them.
        if edge1_left2right ^ edge2_left2right:
            edge1_rc_world = edge1_rc_world[::-1]

    elif direction == "vertical":
        if (edge1_rc_world[-1][1] - edge1_rc_world[0][1]) < 0:
            edge1_top2bottom = False
        else:
            edge1_top2bottom = True
        if (edge2_rc_world[-1][1] - edge2_rc_world[0][1]) < 0:
            edge2_top2bottom =  False
        else:
            edge2_top2bottom = True

        # If the lines do not have the same direction, reverse one of them.
        if edge1_top2bottom ^ edge2_top2bottom:
            edge2_rc_world = edge2_rc_world[::-1]

    """

    if quadrant in ["topBottom", "bottomTop"]:

        # Get search range
        edge1_rc_world_x = edge1_rc_world[:, 0]
        edge2_rc_world_x = edge2_rc_world[:, 0]
        minX = np.min([np.min(edge1_rc_world_x), np.min(edge2_rc_world_x)])
        maxX = np.max([np.max(edge1_rc_world_x), np.max(edge2_rc_world_x)])

        # Pre-allocate lists for later use
        edge1_rc_sampleIdxs = []
        edge2_rc_sampleIdxs = []

        # Loop over x to find where x occurs in edge
        for x in np.arange(minX, maxX, parameters["sampling_deltas"][parameters["iteration"]]):

            # Check whether x value exists in edge1
            indices_edge1 = np.where(x == np.array(edge1_rc_world)[:, 0])[0]
            idxSet_edge1 = indices_edge1.astype("int")
            idx_x_exists = len(idxSet_edge1)>0

            # If a point is found, get corresponding row
            if idx_x_exists:
                rowVals_edge1 = np.array(edge1_rc_world)[idxSet_edge1, 1]

                # Get index corresponding to min/max row value
                if quadrant == "topBottom":
                    maxRowIdx_edge1 = np.argmax(rowVals_edge1)
                    idx_edge1 = idxSet_edge1[maxRowIdx_edge1]

                elif quadrant == "bottomTop":
                    minRowIdx_edge1 = np.argmin(rowVals_edge1)
                    idx_edge1 = idxSet_edge1[minRowIdx_edge1]

                # Populate list with index
                edge1_rc_sampleIdxs.append(idx_edge1)

            # If no point is found, populate list with nan
            else:
                edge1_rc_sampleIdxs.append(np.nan)

            # Repeat for edge 2
            indices_edge2 = np.where(x == np.array(edge2_rc_world)[:, 0])[0]
            idxSet_edge2 = indices_edge2.astype("int")
            idx_x_exists = len(idxSet_edge2)>0

            # If a point is found, get corresponding row
            if idx_x_exists:
                rowVals_edge2 = np.array(edge2_rc_world)[idxSet_edge2, 1]

                # Get index corresponding to min/max row value
                if quadrant == "topBottom":
                    minRowIdx_edge2 = np.argmin(rowVals_edge2)
                    idx_edge2 = idxSet_edge2[minRowIdx_edge2]

                elif quadrant == "bottomTop":
                    maxRowIdx_edge2 = np.argmax(rowVals_edge2)
                    idx_edge2 = idxSet_edge2[maxRowIdx_edge2]

                # Populate list with index
                edge2_rc_sampleIdxs.append(idx_edge2)

            # If no point is found, populate list with nan
            else:
                edge2_rc_sampleIdxs.append(np.nan)

    elif quadrant in ["leftRight", "rightLeft"]:

        # Get search range
        edge1_rc_world_y = edge1_rc_world[:, 1]
        edge2_rc_world_y = edge2_rc_world[:, 1]
        minY = np.min([np.min(edge1_rc_world_y), np.min(edge2_rc_world_y)])
        maxY = np.max([np.max(edge1_rc_world_y), np.max(edge2_rc_world_y)])

        # Pre-allocate lists for later use
        edge1_rc_sampleIdxs = []
        edge2_rc_sampleIdxs = []

        # Loop over y to find where x occurs in edge
        for y in np.arange(minY, maxY, parameters["sampling_deltas"][parameters["iteration"]]):

            # Start with edge1
            indices_edge1 = np.where(y == np.array(edge1_rc_world)[:, 1])[0]
            idxSet_edge1 = indices_edge1.astype("int")
            idx_y_exists = len(idxSet_edge1)>0

            # If a point is found, get corresponding row
            if idx_y_exists:
                colVals_edge1 = np.array(edge1_rc_world)[idxSet_edge1, 0]

                # Get index corresponding to min/max row value
                if quadrant == "rightLeft":
                    maxColIdx_edge1 = np.argmax(colVals_edge1)
                    idx_edge1 = idxSet_edge1[maxColIdx_edge1]

                elif quadrant == "leftRight":
                    minColIdx_edge1 = np.argmin(colVals_edge1)
                    idx_edge1 = idxSet_edge1[minColIdx_edge1]

                # Populate list with index
                edge1_rc_sampleIdxs.append(idx_edge1)

            # If no point is found, populate list with nan
            else:
                edge1_rc_sampleIdxs.append(np.nan)

            # Repeat for edge 2
            indices_edge2 = np.where(y == np.array(edge2_rc_world)[:, 1])[0]
            idxSet_edge2 = indices_edge2.astype("int")
            idx_y_exists = len(idxSet_edge2)>0

            # If a point is found, get corresponding row
            if idx_y_exists:
                colVals_edge2 = np.array(edge2_rc_world)[idxSet_edge2, 0]

                # Get index corresponding to min/max row value
                if quadrant == "rightLeft":
                    minColIdx_edge2 = np.argmin(colVals_edge2)
                    idx_edge2 = idxSet_edge2[minColIdx_edge2]

                elif quadrant == "leftRight":
                    maxColIdx_edge2 = np.argmax(colVals_edge2)
                    idx_edge2 = idxSet_edge2[maxColIdx_edge2]

                # Populate list with index
                edge2_rc_sampleIdxs.append(idx_edge2)

            # If no point is found, populate list with nan
            else:
                edge2_rc_sampleIdxs.append(np.nan)

    else:
        raise ValueError("Unexpected quadrant type")

    assert len(edge1_rc_sampleIdxs) == len(edge2_rc_sampleIdxs), "Edge sampleIdxs should be same length!"

    both_x = list(edge1_rc_world[:, 0]) + list(edge2_rc_world[:, 0])

    # Find nan and non-nan values in sample indices
    notNanIdxs_edge1 = [int(i) for i in np.nonzero(~np.isnan(edge1_rc_sampleIdxs))[0]]
    nanIdxs_edge1 = [int(j) for j in np.nonzero(np.isnan(edge1_rc_sampleIdxs))[0]]

    notNanIdxs_edge2 = [int(i) for i in np.nonzero(~np.isnan(edge2_rc_sampleIdxs))[0]]
    nanIdxs_edge2 = [int(j) for j in np.nonzero(np.isnan(edge2_rc_sampleIdxs))[0]]

    # Preallocate edge variables
    edge1_rc_world_sampled = np.zeros((len(edge1_rc_sampleIdxs), 2))
    edge2_rc_world_sampled = np.zeros((len(edge2_rc_sampleIdxs), 2))

    # Populate edge of real world coordinates based on found indices
    edge1_rc_samplenotnan = [edge1_rc_sampleIdxs[k] for k in notNanIdxs_edge1]
    edge2_rc_samplenotnan = [edge2_rc_sampleIdxs[k] for k in notNanIdxs_edge2]

    # List comprehension is required to handle both ints and list of ints
    edge1_rc_filling = [edge1_rc_world[m] for m in edge1_rc_samplenotnan]
    edge2_rc_filling = [edge2_rc_world[m] for m in edge2_rc_samplenotnan]

    if notNanIdxs_edge1:
        edge1_rc_world_sampled[notNanIdxs_edge1, :] = edge1_rc_filling

    if notNanIdxs_edge2:
        edge2_rc_world_sampled[notNanIdxs_edge2, :] = edge2_rc_filling

    edge1_rc_world_sampled[nanIdxs_edge1] = np.nan
    edge2_rc_world_sampled[nanIdxs_edge2] = np.nan

    # Pre-allocate hist variables
    hists1_sampled = np.zeros((len(edge1_rc_sampleIdxs), np.shape(histograms_a)[-1]))
    hists2_sampled = np.zeros((len(edge2_rc_sampleIdxs), np.shape(histograms_b)[-1]))

    # Populate hists based on found indices
    hists1_filling = np.array([histograms_a[n] for n in edge1_rc_samplenotnan])
    hists2_filling = np.array([histograms_b[n] for n in edge2_rc_samplenotnan])

    hists1_sampled[notNanIdxs_edge1, :] = hists1_filling
    hists2_sampled[notNanIdxs_edge2, :] = hists2_filling

    hists1_sampled[nanIdxs_edge1] = np.nan
    hists2_sampled[nanIdxs_edge2] = np.nan

    overhangIdxs = np.nonzero(nanIdxs_edge2)

    # Compute cost as long as it is defined
    if len(edge1_rc_samplenotnan)>0 and len(edge2_rc_samplenotnan)>0:
        intensityCosts = np.nansum((hists1_sampled-hists2_sampled)**2)
    else:
        intensityCosts = np.nan

    """

    ### Determine which point is inner point and which is outer
    line1_pt1 = edge_tform_theilsen_a[0, :]
    line1_pt2 = edge_tform_theilsen_a[-1, :]
    line1_pts = [line1_pt1, line1_pt2]

    line2_pt1 = edge_tform_theilsen_b[0, :]
    line2_pt2 = edge_tform_theilsen_b[-1, :]
    line2_pts = [line2_pt1, line2_pt2]

    # Calculate distance from center of mass
    line1_distsFromCoM = [np.linalg.norm(line1_pt1 - parameters["center_of_mass"]),
                          np.linalg.norm(line1_pt2 - parameters["center_of_mass"])]
    line2_distsFromCoM = [np.linalg.norm(line2_pt1 - parameters["center_of_mass"]),
                          np.linalg.norm(line2_pt2 - parameters["center_of_mass"])]

    # Get indices of inner and outer points of both lines
    innerPtIdx_line1 = np.argmin(line1_distsFromCoM)
    outerPtIdx_line1 = 1 if innerPtIdx_line1==0 else 0

    innerPtIdx_line2 = np.argmin(line2_distsFromCoM)
    outerPtIdx_line2 = 1 if innerPtIdx_line2==0 else 0

    # Get the inner and outer points
    line1_innerPt = line1_pts[innerPtIdx_line1]
    line1_outerPt = line1_pts[outerPtIdx_line1]
    line2_innerPt = line2_pts[innerPtIdx_line2]
    line2_outerPt = line2_pts[outerPtIdx_line2]

    # Compute overlapAndUnderlapCost as sum of distances between extrema
    innerPointWeight = 1 - parameters["outer_point_weight"]
    innerPointNorm = np.linalg.norm(line1_innerPt - line2_innerPt)**2
    outerPointNorm = np.linalg.norm(line1_outerPt - line2_outerPt)**2
    overlapAndUnderlapCosts = innerPointWeight * innerPointNorm + parameters["outer_point_weight"] * outerPointNorm

    intensityCosts = 0

    return intensityCosts, overlapAndUnderlapCosts