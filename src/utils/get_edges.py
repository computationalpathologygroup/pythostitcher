import numpy as np
import cv2
from skimage import morphology
from matplotlib import pyplot as plt
from .get_boundary import get_boundary
from .min_bbox import min_bbox


def process_bbox_vals(bbox):
    """
    Custom function to compute starting point +  width/height of bbox coordinates.

    input:
    - cv2 bbox consisting of center coordinate, width, height and angle

    output:
    - bbox upperleft corner, width, height and angle
    """

    width = bbox[1][0]
    height = bbox[1][1]
    angle_a = bbox[2]

    diag = np.sqrt(width ** 2 + height ** 2)

    angle_b = math.atan(height / width)
    angle_c = math.radians(angle_a) + angle_b

    new_height = math.sin(angle_c) * 0.5 * diag
    new_width = math.cos(angle_c) * 0.5 * diag

    start = [bbox[0][0] - new_width, bbox[0][1] - new_height]

    return start, width, height, angle_a


def get_box_corners(bbox, contour):
    """
    Custom function to obtain coordinates of corner A and corner C. Corner A is the
    corner of the bounding box which represents the center of the prostate. Corner C
    is the corner of the bounding box which represents the corner furthest away
    from corner A. Corners are named in clockwise direction.

    Example: upperleft quadrant
    C  >  D
    ^     v
    B  <  A

    """

    # Convert bbox object to corner points. These corner points are always oriented counter clockwise.
    box_corners = cv2.boxPoints(bbox)

    # Get list of x-y values of contour
    x_points = [i[0] for i in contour]
    y_points = [i[1] for i in contour]
    distances = []
    mask_corners = []

    # Compute smallest distance from each corner point to any point in contour
    for corner in box_corners:
        dist_x = [np.abs(corner[0] - x_point) for x_point in x_points]
        dist_y = [np.abs(corner[1] - y_point) for y_point in y_points]
        dist = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(dist_x, dist_y)]
        mask_corners.append(np.argmin(dist))
        distances.append(np.min(dist))

    # Corner c should always be the furthest away from the mask
    corner_c_idx = np.argmax(distances)
    corner_c = box_corners[corner_c_idx]

    # Corner a is the opposite corner and is found 2 indices further
    corner_idxs = [0, 1, 2, 3] * 2
    corner_a_idx = corner_idxs[corner_c_idx + 2]
    corner_a = box_corners[corner_a_idx]

    # Corner b corresponds to the corner 1 index before corner c
    corner_b_idx = corner_idxs[corner_c_idx - 1]
    corner_b = box_corners[corner_b_idx]

    # Corner d corresponds to the corner 1 index after corner c
    corner_d_idx = corner_idxs[corner_c_idx + 1]
    corner_d = box_corners[corner_d_idx]

    named_corners = [corner_a, corner_b, corner_c, corner_d]

    return named_corners


def compute_edges(box_corners, contour):
    """
    Function to get the point on the mask that is closed to the corner point from the bounding box
    """

    # Get list of x-y values of contour
    x_points = [i[0] for i in contour]
    y_points = [i[1] for i in contour]
    mask_corner_idxs = []
    mask_corners = []

    # Compute the closest point on the contour for each named corner
    for corner in box_corners:
        dist_x = [np.abs(corner[0] - x_point) for x_point in x_points]
        dist_y = [np.abs(corner[1] - y_point) for y_point in y_points]
        dist = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(dist_x, dist_y)]
        mask_corner_idx = np.argmin(dist)
        mask_corner_idxs.append(mask_corner_idx)
        mask_corners.append(contour[mask_corner_idx])

    # Get indices from edge AB. Account for orientation of the contour
    if mask_corner_idxs[0] < mask_corner_idxs[1]:
        edge_AB = list(contour[mask_corner_idxs[0]:mask_corner_idxs[1]])
    else:
        edge_AB = list(contour[mask_corner_idxs[0]:]) + list(contour[:mask_corner_idxs[1]])

    # Get indices from edge AD. Account for orientation of the contour
    if mask_corner_idxs[3] < mask_corner_idxs[0]:
        edge_AD = list(contour[mask_corner_idxs[3]:mask_corner_idxs[0]])
    else:
        edge_AD = list(contour[mask_corner_idxs[3]:]) + list(contour[:mask_corner_idxs[0]])

    # Flip edge AD so that it runs from center to capsule
    edge_AD = edge_AD[::-1]

    return mask_corners, np.array(edge_AB), np.array(edge_AD)


def get_edges(mask, fragment):

    # Get contour of the mask
    mask = mask.astype(np.uint8)
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Ensure contour is oriented clockwise
    contour = np.squeeze(contour[0])[::-1]

    # Get bounding box around contour
    bbox = cv2.minAreaRect(contour)

    # Get list with corners from A -> D
    bbox_corners = get_box_corners(bbox, contour)

    # Get edge AB and AD
    _, edge_AB, edge_AD = compute_edges(bbox_corners, contour)

    # Return a list with the horizontal and vertical edge of the fragment, which depends on orientation
    if fragment == "A":
        h_edge = edge_AB
        v_edge = edge_AD
    elif fragment == "B":
        h_edge = edge_AD
        v_edge = edge_AB
    elif fragment == "C":
        h_edge = edge_AD
        v_edge = edge_AB
    elif fragment == "D":
        h_edge = edge_AB
        v_edge = edge_AD
    else:
        raise ValueError("Fragment must be one of A/B/C/D")

    return h_edge, v_edge
