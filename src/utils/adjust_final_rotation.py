import cv2
import numpy as np

from .transformations import warp_image


def adjust_final_rotation(image):
    """
    Custom function to compensate the slight rotation that might occur during the
    genetic algorithm.

    Input:
        - Image of all stitched quadrants

    Output:
        - Rotated image of all stitched quadrants
    """

    # Obtain mask
    mask = (image.astype("uint8")[:, :, 0] > 0) * 255

    # Get largest contour
    cnt, _ = cv2.findContours(
        mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    cnt = np.squeeze(max(cnt, key=cv2.contourArea))

    # Compute bounding box around contour
    bbox = cv2.minAreaRect(cnt)

    # Adjust angle
    angle = bbox[2]
    if angle > 45:
        angle = 90 - angle

    # Get center of contour
    moment = cv2.moments(cnt)
    center_x = int(moment["m10"] / moment["m00"])
    center_y = int(moment["m01"] / moment["m00"])

    # Get rotated image
    final_image = warp_image(
        src=image, center=(center_x, center_y), translation=(0, 0), rotation=-angle
    )

    return final_image
