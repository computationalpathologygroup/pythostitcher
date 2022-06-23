import numpy as np


def get_largest_contour(contours):
    """
    Custom function to return the largest contour in the cv2 findContours function.
    Although the findContours function should only find one contour if the tissue
    segmentation mask is correct, this function might catch some errors in case of
    a slightly inaccurate tissue segmentation mask.

    Input:
        - List of contours from cv2 findContours function

    Output:
        - Largest contour
    """

    # Ensure 2D contours
    contours = [np.squeeze(c) for c in contours]

    # Check whether multiple contours exist
    if len(contours) > 1:

        # Get length of each contour
        contour_lengths = []
        for contour in contours:
            length = len(contour)
            contour_lengths.append(length)

        # Get the largest contour
        largest_idx = np.argmax(contour_lengths)
        largest_contour = contours[largest_idx]

        return largest_contour

    # In case of only one contour return this contour
    return contours[0]
