import glob


def get_filename(dirpath, pattern, revName):
    """
    Function to obtain the name of the .mat file in the directory

    :param dirpath: directory with files
    :param pattern: dummy variable
    :param revName: dummy variable
    :return: name without .mat extension
    """

    filedir = glob.glob(dirpath + "/*")
    filenames = [i.split("/")[-1] for i in filedir]
    filematches = [i for i in filenames if i.endswith(".npy")]

    if len(filematches) == 1:
        # Remove .npy extension
        newname = filematches[0].replace(".npy", "")
    else:
        raise ValueError(f"Found {len(filematches)} files with .npy extension, expected 1")

    return newname
