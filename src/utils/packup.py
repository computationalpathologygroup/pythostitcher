

def packup_edges(edge1A_rc, edge2A_rc, edge1B_rc, edge2B_rc, edge1C_rc, edge2C_rc, edge1D_rc, edge2D_rc):

    """
    Simple function to make a dictionary with all edge variables
    """

    edge_dict = dict()
    names = ["edge1A_rc", "edge2A_rc", "edge1B_rc", "edge2B_rc", "edge1C_rc", "edge2C_rc", "edge1D_rc", "edge2D_rc"]
    values = [edge1A_rc, edge2A_rc, edge1B_rc, edge2B_rc, edge1C_rc, edge2C_rc, edge1D_rc, edge2D_rc]

    for name, value in zip(names, values):
        edge_dict[str(name)] = value

    return edge_dict


def packup_images(imgA_t, imgB_t, imgC_t, imgD_t):

    """
    Simple function to store image variables in a dictionary

    """

    img_dict = dict()
    names = ["imgA_t", "imgB_t", "imgC_t", "imgD_t"]
    images = [imgA_t, imgB_t, imgC_t, imgD_t]

    for name, image in zip(names, images):
        img_dict[str(name)] = image

    return img_dict


def packup_lines(line1A_xy, line2A_xy, line1B_xy, line2B_xy, line1C_xy, line2C_xy, line1D_xy, line2D_xy):

    """
    Simple function to create a dictionary with line variables

    """

    line_dict = dict()
    names = ["line1A_xy", "line2A_xy", "line1B_xy", "line2B_xy", "line1C_xy", "line2C_xy", "line1D_xy", "line2D_xy"]
    lines = [line1A_xy, line2A_xy, line1B_xy, line2B_xy, line1C_xy, line2C_xy, line1D_xy, line2D_xy]

    for name, line in zip(names, lines):
        line_dict[str(name)] = line

    return line_dict
