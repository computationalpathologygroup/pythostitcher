import numpy as np
from skimage import transform
from .spatial_ref_object import spatial_ref_object


def get_tformed_images(*args):

    varargin = len(args)

    if varargin == 5:
        imgA_g, imgB_g, imgC_g, imgD_g, tform = args
        rA = spatial_ref_object(imgA_g)
        rB = spatial_ref_object(imgB_g)
        rC = spatial_ref_object(imgC_g)
        rD = spatial_ref_object(imgD_g)
    elif varargin == 9:
        imgA_g, imgB_g, imgC_g, imgD_g, tform, rA, rB, rC, rD = args
    else:
        raise ValueError("Number of input arguments should be either 5 or 9")

    x = np.round(tform)

    imgA_g = transform.rotate(imgA_g, -x[2])
    A = np.array([[1, 0, 0], [0, 1, 0], [x[0], x[1], 1]])
    rA_t = rA
    imgA_t = transform.warp(imgA_g, A)

    imgB_g = transform.rotate(imgB_g, -x[5])
    B = np.array([[1, 0, 0], [0, 1, 0], [x[3], x[4], 1]])
    rB_t = rB
    imgB_t = transform.warp(imgB_g, B)

    imgC_g = transform.rotate(imgC_g, -x[8])
    C = np.array([[1, 0, 0], [0, 1, 0], [x[6], x[7], 1]])
    rC_t = rC
    imgC_t = transform.warp(imgC_g, C)

    imgD_g = transform.rotate(imgD_g, -x[11])
    D = np.array([[1, 0, 0], [0, 1, 0], [x[9], x[10], 1]])
    rD_t = rD
    imgD_t = transform.warp(imgD_g, D)

    return imgA_t, rA_t, imgB_t, rB_t, imgC_t, rC_t, imgD_t, rD_t
