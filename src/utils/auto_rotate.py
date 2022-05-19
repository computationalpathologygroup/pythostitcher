import numpy as np
import cv2
import math

from skimage import morphology
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage import transform

from .get_boundary import get_boundary
from .min_bbox import min_bbox


def process_bbox_vals(bbox):
    """
    Function to compute starting point +  width/height of bbox coordinates

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


def get_corner_ad(bbox, contour):
    """
    Custom function to obtain coordinates of corner A and corner D. Corner A is the
    corner of the bounding box which represents the center of the prostate. Corner D
    is the corner of the bounding box which represents the corner furthest away
    from corner A.
    """

    # Convert bbox to corner points
    box_corners = cv2.boxPoints(bbox)

    # Get list of x-y values of contour
    x_points = [i[0][0] for i in contour[0]]
    y_points = [i[0][1] for i in contour[0]]
    distances = []

    # Compute smallest distance from each corner point to any point in contour
    for corner in box_corners:
        dist_x = [np.abs(corner[0] - x_point) for x_point in x_points]
        dist_y = [np.abs(corner[1] - y_point) for y_point in y_points]
        dist = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(dist_x, dist_y)]
        distances.append(np.min(dist))

    # Corner d should always be the furthest away from the mask
    corner_d_idx = np.argmax(distances)
    corner_d = box_corners[corner_d_idx]

    # Corner a is the opposite corner and is found 2 indices further
    corner_idxs = [0, 1, 2, 3] * 2
    corner_a_idx = corner_idxs[corner_d_idx + 2]
    corner_a = box_corners[corner_a_idx]

    return corner_a, corner_d


def auto_rotate(mask, plot=True):
    """
    Custom function to automatically rotate the input fragments

    Inputs:
    - tissue mask (np.array)
    - name of the fragment [A, B, C, D] (string)

    Outputs:
    - rotation angle
    - bounding box points
    - corner a coordinates

    """

    # Convert mask to grayscale if this has not been done yet
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = mask.astype(np.uint8)

    # Compute boundary of the mask
    cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the encompassing bounding box with the smallest area
    box = cv2.minAreaRect(cnt[0])

    # Get corners of the bbox
    box_corners = cv2.boxPoints(box)

    # Convert the bbox object to bbox parameters for easy plotting
    start, w, h, angle = process_bbox_vals(box)

    # Get corner A
    corner_a, _ = get_corner_ad(box, cnt)

    # Default rotation is ccw. When angle exceeds 45 degrees this will erroneously
    # rotate the orientation of the mask. When specifying a negative angle we
    # can rotate cw.
    if angle > 45:
        true_angle = -(90 - angle)

    # Show original image with bbox and rotated result
    if plot:
        plt.figure()
        plt.subplot(121)
        plt.title("Original mask with bbox")
        plt.imshow(mask, cmap="gray")
        plt.scatter(corner_a[0], corner_a[1], facecolor="r", s=100)
        rect = Rectangle(start, w, h, linewidth=3, edgecolor='r', angle=angle, facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)

        im_r = transform.rotate(mask, true_angle)

        plt.subplot(122)
        plt.title("Rotated mask")
        plt.imshow(im_r, cmap="gray")
        plt.show()

    return true_angle, box_corners, corner_a

    # x = outline_A[:, 1]
    # y = np.shape(mask)[0]+1 - outline_A[:, 0]
    # bbA = min_bbox([x, y])
    # cols = bbA[0, :]
    # rows = np.shape(mask)[0]+1 - bbA[1, :]
    # bboxPts = [np.transpose(cols), np.transpose(rows)]
    #
    # if plotsOn:
    #     plt.figure()
    #     plt.title("Initial mask")
    #     plt.subplot(121)
    #     plt.imshow(mask)
    #     plt.subplot(122)
    #     plt.plot(cols, rows)  #### PERHAPS CHANGE ORDER
    #
    # distances = np.zeros((1, 4))
    # cornerPts = []
    # for k in range(4):
    #     r, c = np.argwhere(mask > 0)
    #     dvals = np.sqrt((r-rows[k]) ** 2 + (c-cols[k]) ** 2)
    #     idx = np.where(dvals == np.min(dvals))
    #     if plotsOn:
    #         plt.figure()
    #         plt.plot(c[idx], r[idx])
    #         distances[k] = np.min(dvals)
    #         cornerPts = [cornerPts, c[idx], r[idx]]
    #
    #     idx = np.where(distances == np.max(distances))
    #     cornerA = [rows[k], cols[idx]]
    #
    #     if plotsOn:
    #         plt.figure()
    #         plt.plot(cornerA[1], cornerA[0])
    #
    #     distances = np.zeros((1, 4))
    #
    #     for m in range(4):
    #         distances[m] = np.sqrt((cornerA[0]-rows[m])**2+(cornerA[1]-cols[m])**2)
    #
    #     idx = np.argsort(distances)
    #     bcCandidates = sorted(distances)
    #     bcCandidates = [rows[idx[1]], cols[idx[1]], rows[idx[2]], cols[idx[2]], rows[idx[3]], cols[idx[3]]]
    #
    #     if plotsOn:
    #         plt.figure()
    #         plt.plot(bcCandidates[1], bcCandidates[0])
    #         plt.plot(bcCandidates[3], bcCandidates[2])
    #     cornerD = [bcCandidates[4], bcCandidates[5]]
    #
    #     cornerA_x = cornerA[1]
    #     cornerA_y = np.shape(mask)[0]+1 - cornerA[0]
    #     bc1_x = bcCandidates[1]
    #     bc1_y = np.shape(mask)[0]+1 - bcCandidates[0]
    #     bc2_x = bcCandidates[3]
    #     bc2_y = np.shape(mask)[0]+1 - bcCandidates[2]
    #     a_bc1 = [bc1_x - cornerA_x, bc1_y - cornerA_y, 0]
    #     a_bc2 = [bc2_x - cornerA_x, bc2_y - cornerA_y, 0]
    #     cp = np.cross(a_bc1, a_bc2)
    #
    #     if np.sum(cp) > 0:
    #         cornerC = [bcCandidates[0], bcCandidates[1]]
    #         cornerB = [bcCandidates[2], bcCandidates[3]]
    #     elif np.sum(cp) < 0:
    #         cornerC = [bcCandidates[2], bcCandidates[3]]
    #         cornerB = [bcCandidates[0], bcCandidates[1]]
    #     else:
    #         raise ValueError('Unexpected results for cross product')
    #
    #     cornerA_x = cornerA[1]
    #     cornerA_y = np.shape(mask)[0]+1 - cornerA[0]
    #     cornerB_x = cornerB[1]
    #     cornerB_y = np.shape(mask)[0]+1 - cornerB[0]
    #     cornerC_x = cornerC[1]
    #     cornerC_y = np.shape(mask)[0]+1 - cornerC[0]
    #     cornerD_x = cornerD[1]
    #     cornerD_y = np.shape(mask)[0]+1 - cornerD[0]
    #
    # if fragment == "A":
    #     x = cornerD_x - cornerC_x
    #     y = cornerD_y - cornerC_y
    #     lengthCD = np.sqrt(x**2+y**2)
    #     x = x/lengthCD
    #     y = y/lengthCD
    # elif fragment == "B":
    #     x = cornerB_x - cornerD_x
    #     y = cornerB_y - cornerD_y
    #     lengthDB = np.sqrt(x ** 2 + y ** 2)
    #     x = x / lengthDB
    #     y = y / lengthDB
    # elif fragment == "C":
    #     x = cornerD_x - cornerB_x
    #     y = cornerD_y - cornerB_y
    #     lengthBD = np.sqrt(x ** 2 + y ** 2)
    #     x = x / lengthBD
    #     y = y / lengthBD
    # elif fragment == "D":
    #     x = cornerC_x - cornerD_x
    #     y = cornerC_y - cornerD_y
    #     lengthDC = np.sqrt(x ** 2 + y ** 2)
    #     x = x / lengthDC
    #     y = y / lengthDC
    # else:
    #     raise ValueError('Invalid fragment input')
    #
    # angle = -(np.arctan2(y, x) * 180 / np.pi)
    # mask_rotated = transform.rotate(mask, angle) ##### MIGHT REQUIRE EXTRA STEP FOR OUTPUT SIZE
    #
    # if plotsOn:
    #     plt.figure()
    #     plt.imshow(mask_rotated)
    #     plt.title(f"Rotated mask with {angle} degrees")

    # return angle, bboxPts, cornerPts