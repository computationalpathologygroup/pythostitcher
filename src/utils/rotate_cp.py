import numpy as np
import math


def rotate_cp(imgCenter_xy, CPs_xy, theta):

    # Compute center of points
    v = np.transpose(CPs_xy)
    imgCenter_xy = np.array(imgCenter_xy)[:, np.newaxis]
    center = np.tile(imgCenter_xy, [1, v.shape[1]])

    # Define transformation matrix
    R = [[math.degrees(math.cos(theta)), -math.degrees(math.sin(theta))],
         [math.degrees(math.sin(theta)), math.degrees(math.cos(theta))]]

    # Apply transformation matrix
    rotated_points = np.dot(np.transpose(v-center), np.array(R).T)
    rotated_points = rotated_points + np.transpose(center)

    return rotated_points


def rotate_2dpoints(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)
