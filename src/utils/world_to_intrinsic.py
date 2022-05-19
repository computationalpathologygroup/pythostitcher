import numpy as np


def world_to_intrinsic(r, xWorld, yWorld):
    """
    Function to convert world coordinates to intrinsic coordinates

    :param r: list with 6 values: XILim, YILim, XWLim, YWLim, PixX, PixY
    :param xWorld: world coordinates X
    :param yWorld: world coordinates Y
    :return: intrinsic coordinates x, intrinsic coordinates y
    """

    assert np.size(xWorld) == np.size(yWorld), "xworld and yworld must have same size"
    assert len(r) == 6, "r requires 6 input arguments: XILim, YILim, XWLim, YWLim, PixX, PixY"

    xIntlim = r[0]
    yIntlim = r[1]
    xWlim = r[2]
    yWlim = r[3]

    xInt = xIntlim[0] + (xWorld - xWLim[0])/r[4]
    yInt = yIntlim[0] + (yWorld - xWLim[0])/r[5]

    return xInt, yInt
