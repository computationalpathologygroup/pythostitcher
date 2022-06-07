import numpy as np


def get_largest_contour(contours):
    """
    Custom function to return the largest contour in the cv2 findContours function.
    """

    # Preprocess
    contours = [np.squeeze(c) for c in contours]

    # Check whether multiple contours exist
    if len(contours) > 1:

        # Get length of each contour
        contour_lengths = []
        for i in range(len(contours)):
            length = len(contours[i])
            contour_lengths.append(length)

        # Get the largest contour
        largest_idx = np.argmax(contour_lengths)
        largest_contour = contours[largest_idx]

        return largest_contour

    # In case of only 1 contour do nothing
    else:
        return contours[0]