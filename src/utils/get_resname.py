
def get_resname(res):
    """
    Custom function to convert the resolution fraction to a string. This is required for creating directories
    for each resolution.

    Input:
        - Resolution

    Output:
        - Resolution name
    """

    assert res<=1, "resolution fraction must be equal to or smaller than the original image"

    resname = "res" + str(int(res*100)).zfill(3)

    return resname
