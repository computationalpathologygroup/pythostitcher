import numpy as np
from sklearn.linear_model import TheilSenRegressor
from .unpack_edges import unpack_edges
from .packup import packup_lines


def get_edge_length(edge):
    """
    Function to compute edge length

    :param edge:
    :return: length
    """

    return np.linalg.norm(edge[0, :]-edge[-1, :])

"""
def get_edge_subset(edge_rc, subset_length, which_edge):

    subsetStartIdx = np.shape(edge_rc)[0] - 1
    subsetStart = edge_rc[subsetStartIdx, :]

    if which_edge in ["edge1A_rc", "edge1C_rc"]:
        subsetEndColumn = subsetStart[1]
        subsetEndIdxs = np.nonzero(np.isclose(edge_rc[:, 1], subsetEndColumn, atol=1e-3))

    elif which_edge in ["edge2A_rc", "edge2B_rc"]:
        subsetEndRow = subsetStart[0]
        subsetEndIdxs = np.nonzero(np.isclose(edge_rc[:, 0], subsetEndRow, atol=1e-3))

    elif which_edge in ["edge1B_rc", "edge1D_rc"]:
        subsetEndColumn = subsetStart[1]
        subsetEndIdxs = np.nonzero(np.isclose(edge_rc[:, 1], subsetEndColumn, atol=1e-3))

    elif which_edge in ["edge2C_rc", "edge2D_rc"]:
        subsetEndRow = subsetStart[0]
        subsetEndIdxs = np.nonzero(np.isclose(edge_rc[:, 0], subsetEndRow, atol=1e-3))

    else:
        raise ValueError("Unexpected edge type, must be in format edge(1/2)(A/B/C/D)_rc")

    # SubsetEndIdxs can either be a (n,) list of indices or one single index.
    try:
        subsetEndIdxs = np.squeeze(subsetEndIdxs)
        subsetEndIdx = subsetEndIdxs[0]
    except:
        subsetEndIdx = np.asscalar(subsetEndIdxs)

    if subsetEndIdx != subsetStartIdx:
        edges_rc_s = edge_rc[subsetEndIdx:subsetStartIdx, :]
    else:
        edges_rc_s = edge_rc[subsetEndIdx]

    return edges_rc_s
"""

 ###############################################################################

def get_edge_subset(edge_rc, subset_length, which_edge):
    """
    Custom function to retrieve a subset of the edge
    """

    subset_startidx = np.shape(edge_rc)[0] - 1
    subset_start = edge_rc[subset_startidx, :]

    if which_edge in ["edgeA_h", "edgeC_h"]:
        subset_endcolumn = subset_start[1] + np.round(subset_length)
        subset_endidx = np.nonzero(np.isclose(edge_rc[:, 1], subset_endcolumn, atol=1e-3))

    elif which_edge in ["edgeA_v", "edgeB_v"]:
        subset_endrow = subset_start[0] + np.round(subset_length)
        subset_endidx = np.nonzero(np.isclose(edge_rc[:, 0], subset_endrow, atol=1e-3))

    elif which_edge in ["edgeB_h", "edgeD_h"]:
        subset_endcolumn = subset_start[1] - np.round(subset_length)
        subset_endidx = np.nonzero(np.isclose(edge_rc[:, 1], subset_endcolumn, atol=1e-3))

    elif which_edge in ["edgeC_v", "edgeD_v"]:
        subset_endrow = subset_start[0] - np.round(subset_length)
        subset_endidx = np.nonzero(np.isclose(edge_rc[:, 0], subset_endrow, atol=1e-3))

    # Obtain edge subset
    if subset_endidx != subset_startidx:
        edges_rc_s = edge_rc[subset_endidx:subset_startidx, :]
    else:
        edges_rc_s = edge_rc[subset_endidx]

    return edges_rc_s


def get_theilsen_lines(quadrant_A, quadrant_B, quadrant_C, quadrant_D, parameters):
    """
    Custom function to obtain the Theilsen lines from the quadrants' edges
    """

    # Get all edges and compute edge length
    all_edges = [quadrant_A.h_edge, quadrant_B.v_edge, quadrant_B.h_edge, quadrant_B.v_edge,
                 quadrant_C.h_edge, quadrant_C.v_edge, quadrant_D.h_edge, quadrant_D.v_edge]
    which_edges = np.array([["edgeA_h", "edgeC_h"], ["edgeA_v", "edgeB_v"],
                            ["edgeB_h", "edgeD_h"], ["edgeC_v", "edgeD_v"]])

    edge_lengths = [np.linalg.norm(edge[0, :]-edge[-1, :]) for edge in all_edges]
    mean_edge_length = np.mean(edge_lengths)
    subset_length = parameters["fraction_edge_length"] * mean_edge_length

    # Create some variables for use in loop
    edges1 = [quadrant_A.h_edge, quadrant_A.v_edge, quadrant_B.h_edge, quadrant_C.v_edge]
    edges2 = [quadrant_C.h_edge, quadrant_B.v_edge, quadrant_D.h_edge, quadrant_D.v_edge]
    edges1_rc_s = []
    edges2_rc_s = []

    # Obtain subset for all edges
    for i in range(4):
        edges1_result = get_edge_subset(edges1[i], subset_length, which_edges[i][0])
        edges1_rc_s.append(edges1_result)
        edges2_result = get_edge_subset(edges2[i], subset_length, which_edges[i][1])
        edges2_rc_s.append(edges2_result)

    # Create variables for next loop
    edges1 = edges1_rc_s
    edges2 = edges2_rc_s
    lines1 = []
    lines2 = []
    directions = ["top_bottom", "left_right", "bottom_top", "right_left"]

    for count, loc in enumerate(directions):
        edge1_rc = edges1[count]
        edge2_rc = edges2[count]

        if loc in ["top_bottom", "bottom_top"]:

            # Initiate first Theilsen instance
            TheilSen1 = TheilSenRegressor()
            x1 = [e[0] for e in edge1_rc]
            y1 = [e[1] for e in edge1_rc]

            # Fit coordinates to line and predict new line
            Theilsen1.fit(x1, y1)
            line1_xlimits = [np.min(x1), np.max(y1)]
            line1_x = [line1_xlimits[0], line1_xlimits[1]]
            line1_x = np.array(line1_x)[:, np.newaxis]
            line1_y = TheilSen1.predict(line1_x)

            # Initiate second Theilsen instance
            TheilSen2 = TheilSenRegressor()
            x2 = [e[0] for e in edge2_rc]
            y2 = [e[1] for e in edge2_rc]

            # Fit coordinates to line and predict new line
            TheilSen2.fit(x2, y2)
            line2_xlimits = [np.min(x2), np.max(y2)]
            line2_x = [line2_xlimits[0], line2_xlimits[1]]
            line2_x = np.array(line2_x)[:, np.newaxis]
            line2_y = TheilSen2.predict(line2_x)

        elif loc in ["left_right", "right_left"]:

            # Initiate first Theilsen instance
            TheilSen1 = TheilSenRegressor()
            x1 = [e[0] for e in edge1_rc]
            y1 = [e[1] for e in edge1_rc]

            # Fit coordinates to line and predict new line
            TheilSen1.fit(x1, y1.ravel())
            line1_xlimits = [np.min(x1), np.max(y1)]
            line1_y = [line1_xlimits[0], line1_xlimits[1]]
            line1_y = np.array(line1_y)[:, np.newaxis]
            line1_x = TheilSen1.predict(line1_y)

            # Initiate second Theilsen instance
            TheilSen2 = TheilSenRegressor()
            x2 = [e[0] for e in edge2_rc]
            y2 = [e[1] for e in edge2_rc]

            # Fit coordinates to line and predict new line
            TheilSen2.fit(x2, y2.ravel())
            line2_xlimits = [np.min(x2), np.max(y2)]
            line2_y = [line2_xlimits[0], line2_xlimits[1]]
            line2_y = np.array(line2_y)[:, np.newaxis]
            line2_x = TheilSen2.predict(line2_y)

        #
        lines1.append([line1_x, line1_y])
        lines2.append([line2_x, line2_y])

    return None

"""
def get_theilsen_lines(edgesStruct, fractionOfEdgeLengthToUse):
    edge1A_rc, edge2A_rc, edge1B_rc, edge2B_rc, edge1C_rc, edge2C_rc, edge1D_rc, edge2D_rc = unpack_edges(edgesStruct)
    which_edges = np.array([["edge1A_rc", "edge1C_rc"], ["edge2A_rc", "edge2B_rc"], ["edge1B_rc", "edge1D_rc"], ["edge2C_rc", "edge2D_rc"]])

    edges1 = [edge1A_rc, edge2A_rc, edge1B_rc, edge2C_rc]
    edges2 = [edge1C_rc, edge2B_rc, edge1D_rc, edge2D_rc]

    allEdges = [edge1A_rc, edge2A_rc, edge1B_rc, edge2C_rc, edge1C_rc, edge2B_rc, edge1D_rc, edge2D_rc]

    edgeLengths = [get_edge_length(edge) for edge in allEdges]
    meanEdgeLength = np.mean(edgeLengths)
    subset_length = fractionOfEdgeLengthToUse*meanEdgeLength

    edges1_rc_s = []
    edges2_rc_s = []

    for i in range(4):
        edges1_result = get_edge_subset(edges1[i], subset_length, which_edges[i, 0])
        edges1_rc_s.append(edges1_result)
        edges2_result = get_edge_subset(edges2[i], subset_length, which_edges[i, 1])
        edges2_rc_s.append(edges2_result)

    edges1 = edges1_rc_s
    edges2 = edges2_rc_s

    whichQuadrants = ["topBottom", "leftRight", "bottomTop", "rightLeft"]
    lines1 = []
    lines2 = []

    for i in range(4):
        edge1_rc = edges1[i]
        edge2_rc = edges2[i]

        if whichQuadrants[i] in ["topBottom", "bottomTop"]:

            TheilSen1 = TheilSenRegressor()

            # Account for 1D and 2D lists of points
            if len(edge1_rc.shape) == 2:
                x1 = edge1_rc[:, 1].reshape(-1, 1)
                y1 = edge1_rc[:, 0].reshape(-1, 1)
            elif len(edge1_rc.shape) == 1:
                x1 = edge1_rc[1].reshape(-1, 1)
                y1 = edge1_rc[0].reshape(-1, 1)

            TheilSen1.fit(x1, y1.ravel())
            line1_xlimits = [np.min(x1), np.max(y1)]
            line1_x = [line1_xlimits[0], line1_xlimits[1]]
            line1_x = np.array(line1_x)[:, np.newaxis]
            line1_y = TheilSen1.predict(line1_x)

            TheilSen2 = TheilSenRegressor()

            # Account for 1D and 2D lists of points
            if len(edge2_rc.shape) == 2:
                x2 = edge2_rc[:, 1].reshape(-1, 1)
                y2 = edge2_rc[:, 0].reshape(-1, 1)
            elif len(edge2_rc.shape) == 1:
                x2 = edge2_rc[1].reshape(-1, 1)
                y2 = edge2_rc[0].reshape(-1, 1)

            TheilSen2.fit(x2, y2.ravel())
            line2_xlimits = [np.min(x2), np.max(y2)]
            line2_x = [line2_xlimits[0], line2_xlimits[1]]
            line2_x = np.array(line2_x)[:, np.newaxis]
            line2_y = TheilSen2.predict(line2_x)

        elif whichQuadrants[i] in ["leftRight", "rightLeft"]:

            TheilSen1 = TheilSenRegressor()

            # Account for 1D and 2D lists of points
            if len(edge1_rc.shape) == 2:
                x1 = edge1_rc[:, 0].reshape(-1, 1)
                y1 = edge1_rc[:, 1].reshape(-1, 1)
            elif len(edge1_rc.shape) == 1:
                x1 = edge1_rc[0].reshape(-1, 1)
                y1 = edge1_rc[1].reshape(-1, 1)

            TheilSen1.fit(x1, y1.ravel())
            line1_xlimits = [np.min(x1), np.max(y1)]
            line1_y = [line1_xlimits[0], line1_xlimits[1]]
            line1_y = np.array(line1_y)[:, np.newaxis]
            line1_x = TheilSen1.predict(line1_y)

            TheilSen2 = TheilSenRegressor()

            # Account for 1D and 2D lists of points
            if len(edge2_rc.shape) == 2:
                x2 = edge2_rc[:, 0].reshape(-1, 1)
                y2 = edge2_rc[:, 1].reshape(-1, 1)
            elif len(edge2_rc.shape) == 1:
                x2 = edge2_rc[0].reshape(1, -1)
                y2 = edge2_rc[1].reshape(1, -1)

            TheilSen2.fit(x2, y2.ravel())
            line2_xlimits = [np.min(x2), np.max(y2)]
            line2_y = [line2_xlimits[0], line2_xlimits[1]]
            line2_y = np.array(line2_y)[:, np.newaxis]
            line2_x = TheilSen2.predict(line2_y)

        else:
            raise ValueError("Unexpected whichQuadrants type, must be either topbottom/bottomtop/leftright/rightleft")

        # Account for 1D and 2D lists of points
        if len(line1_x.shape) == 2:
            line1_x = np.squeeze(np.transpose(line1_x))
        if len(line1_y.shape) == 2:
            line1_y = np.squeeze(np.transpose(line1_y))
        if len(line2_x.shape) == 2:
            line2_x = np.squeeze(np.transpose(line2_x))
        if len(line2_y.shape) == 2:
            line2_y = np.squeeze(np.transpose(line2_y))

        lines1.append([line1_x, line1_y])
        lines2.append([line2_x, line2_y])

    line1A_xy = np.transpose(lines1[0])
    line1C_xy = np.transpose(lines2[0])
    line2A_xy = np.transpose(lines1[1])
    line2B_xy = np.transpose(lines2[1])
    line1B_xy = np.transpose(lines1[2])
    line1D_xy = np.transpose(lines2[2])
    line2C_xy = np.transpose(lines1[3])
    line2D_xy = np.transpose(lines2[3])

    theilSenLinesStruct = packup_lines(line1A_xy, line2A_xy, line1B_xy, line2B_xy, line1C_xy, line2C_xy, line1D_xy, line2D_xy)

    return theilSenLinesStruct
"""