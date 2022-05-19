import scipy
import numpy as np
import math



def min_bbox(X):
    """
    Compute the minimum bounding box of a set of 2D points

    :param X: matrix of 2D coordinates with n points
    :return: matrix of 2D coordinates of bounding box
    """

    # Compute convex hull
    k = scipy.spatial.ConvexHull(X)
    CH = X[:, k]

    # Compute angle to
    E = np.diff(CH, n=1, axis=1)
    T = math.atan2(E[1, :], E[0, :])
    T = np.unique(T % (math.pi/2))

    Ta = np.reshape(np.matlib.repmat(T, 2, 2), [2*len(T), 2])
    Tb = np.matlib.repmat(([0, -math.pi/2], [math.pi/2, 0]), len(T), 1)
    R = math.cos(Ta+Tb)

    # Rotate CH
    RCH = R*CH

    # Compute border size and area of bbox
    bsize = np.max(RCH, axis=1) - np.min(RCH, axis=1)
    area = np.prod(np.reshape(bsize, [2, len(bsize)/2]))

    # Compute minimal area and index
    i = np.argmin(area)

    # Compute boundaries of rotated frame
    Rf = R[np.add(2*i, [-1, 0]), :]
    bound = Rf * CH
    bmin = np.min(bound, axis=1)
    bmax = np.max(bound, axis=2)

    # Compute corner of bounding box
    Rf = np.array(Rf).transpose()
    bb = np.empty((len(bmax[0]*Rf[:, 0]), 4))
    bb[:, 3] = bmax[0] * Rf[:, 0] + bmin[1] * Rf[:, 1]
    bb[:, 0] = bmin[0] * Rf[:, 0] + bmin[1] * Rf[:, 1]
    bb[:, 1] = bmin[0] * Rf[:, 0] + bmax[1] * Rf[:, 1]
    bb[:, 2] = bmax[0] * Rf[:, 0] + bmax[1] * Rf[:, 1]

    return bb
