import numpy as np


def sub2ind(array_shape, rows, cols):
    """
    Helper function to convert row/col indexing to linear indexing

    :param array_shape: shape of the matrix that should be translated
    :param rows: the rows you want to convert
    :param cols: the columns you want to convert
    :return: list of indices
    """

    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1

    return ind


def get_histograms(localW, localH, e1_keep, lwindowSize, img1_g_tformed, nbins, quadrant):
    """
    Function to get the histograms

    :param localW:
    :param localH:
    :param e1_keep:
    :param lwindowSize:
    :param img1_g_tformed:
    :param nbins:
    :param quadrant:
    :return:
    """

    # Determine window slide based on quadrant type
    if quadrant == "top":
        windowshift_w, windowshift_h = np.meshgrid(np.arange(-localW-1, localW+1), np.arange(-localH-1, 0))
        windowshifts = np.squeeze(np.array([windowshift_h.ravel(), windowshift_w.ravel()]))
    elif quadrant == "bottom":
        windowshift_w, windowshift_h = np.meshgrid(np.arange(-localW-1, localW+1), np.arange(0, localH+1))
        windowshifts = np.squeeze(np.array([windowshift_h.ravel(), windowshift_w.ravel()]))
    elif quadrant == "left":
        windowshift_w, windowshift_h = np.meshgrid(np.arange(-localH-1, 0), np.arange(-localW-1, localW+1))
        windowshifts = np.squeeze(np.array([windowshift_h.ravel(), windowshift_w.ravel()]))
    elif quadrant == "right":
        windowshift_w, windowshift_h = np.meshgrid(np.arange(0, localH+1), np.arange(-localW-1, localW+1))
        windowshifts = np.squeeze(np.array([windowshift_h.ravel(), windowshift_w.ravel()]))
    else:
        raise ValueError("Unexpected quadrant type: must be one on top/bottom/left/right")

    # Compute window shift
    windowshifts = windowshifts[:, :, np.newaxis]
    windowshifts = np.repeat(windowshifts, np.shape(e1_keep)[0], axis=-1)
    windowshifts = np.transpose(windowshifts, [2, 1, 0])

    # Convert indices to subscripts
    e1_keep_3d = e1_keep[:, :, np.newaxis]
    e1_keep_3d = np.repeat(e1_keep_3d, np.shape(windowshifts)[-1], axis=-1)
    e1_patchIndices_local = e1_keep_3d + windowshifts
    e1_patchIndices_local = e1_patchIndices_local.astype("int")
    e1_patchIndices_linIdxs_local = np.unravel_index(e1_patchIndices_local, np.shape(img1_g_tformed))
    tpatches_local = img1_g_tformed[e1_patchIndices_linIdxs_local]

    # Pre allocate histogram matrix
    hists_local = np.empty((np.shape(e1_keep)[0], nbins+1))
    hists_local[:] = np.nan

    # Compute local histogram per patch
    for i in range(np.shape(e1_keep)[0]):
        patch = tpatches_local[i, :]
        tnout, _ = np.histogram(patch, bins=nbins+1, range=[0, 1])
        hists_local[i, :] = tnout/np.sum(tnout)

    return hists_local
