from .get_histograms import get_histograms, sub2ind
import numpy as np


def get_histograms_and_intensities(tformed_images, tformed_edges, simple_hist_size, nbins):
    """
    This function unpacks images and edges to compute histograms and intensities
    """

    assert all([i % 2 != 0 for i in simple_hist_size]), "Window size must be odd"

    # Compute window size
    hist_width = np.floor(simple_hist_size[0] / 2)
    hist_height = np.floor(simple_hist_size[1] - 1)
    hist_windowsize = (hist_height + 1) * (2 * hist_width + 1)

    # Get images
    imgA = tformed_images["imgA_t"]
    imgB = tformed_images["imgB_t"]
    imgC = tformed_images["imgC_t"]
    imgD = tformed_images["imgD_t"]

    # Get edges
    edge1A_rc = tformed_edges["edge1A_rc"]
    edge2A_rc = tformed_edges["edge2A_rc"]
    edge1B_rc = tformed_edges["edge1B_rc"]
    edge2B_rc = tformed_edges["edge2B_rc"]
    edge1C_rc = tformed_edges["edge1C_rc"]
    edge2C_rc = tformed_edges["edge2C_rc"]
    edge1D_rc = tformed_edges["edge1D_rc"]
    edge2D_rc = tformed_edges["edge2D_rc"]

    # Get histograms
    hist_1A = get_histograms(hist_width, hist_height, edge1A_rc, hist_windowsize, imgA, nbins, 'top')
    hist_1C = get_histograms(hist_width, hist_height, edge1C_rc, hist_windowsize, imgC, nbins, 'bottom')
    hist_1D = get_histograms(hist_width, hist_height, edge1D_rc, hist_windowsize, imgD, nbins, 'bottom')
    hist_1B = get_histograms(hist_width, hist_height, edge1B_rc, hist_windowsize, imgB, nbins, 'top')
    hist_2A = get_histograms(hist_width, hist_height, edge2A_rc, hist_windowsize, imgA, nbins, 'left')
    hist_2B = get_histograms(hist_width, hist_height, edge2B_rc, hist_windowsize, imgB, nbins, 'right')
    hist_2D = get_histograms(hist_width, hist_height, edge2D_rc, hist_windowsize, imgD, nbins, 'right')
    hist_2C = get_histograms(hist_width, hist_height, edge2C_rc, hist_windowsize, imgC, nbins, 'left')

    # Convert subscript to linear indices and apply these indices to the flattened array
    edge1A_rc_linIdxs = np.ravel_multi_index((edge1A_rc[:, 0], edge1A_rc[:, 1]), imgA.shape)
    intensities_1A = np.ravel(imgA)[edge1A_rc_linIdxs]
    edge1B_rc_linIdxs = np.ravel_multi_index((edge1B_rc[:, 0], edge1B_rc[:, 1]), imgB.shape)
    intensities_1B = np.ravel(imgB)[edge1B_rc_linIdxs]
    edge1C_rc_linIdxs = np.ravel_multi_index((edge1C_rc[:, 0], edge1C_rc[:, 1]), imgC.shape)
    intensities_1C = np.ravel(imgC)[edge1C_rc_linIdxs]
    edge1D_rc_linIdxs = np.ravel_multi_index((edge1D_rc[:, 0], edge1D_rc[:, 1]), imgD.shape)
    intensities_1D = np.ravel(imgD)[edge1D_rc_linIdxs]

    edge2A_rc_linIdxs = np.ravel_multi_index((edge2A_rc[:, 0], edge2A_rc[:, 1]), imgA.shape)
    intensities_2A = np.ravel(imgA)[edge2A_rc_linIdxs]
    edge2B_rc_linIdxs = np.ravel_multi_index((edge2B_rc[:, 0], edge2B_rc[:, 1]), imgB.shape)
    intensities_2B = np.ravel(imgB)[edge2B_rc_linIdxs]
    edge2C_rc_linIdxs = np.ravel_multi_index((edge2C_rc[:, 0], edge2C_rc[:, 1]), imgC.shape)
    intensities_2C = np.ravel(imgC)[edge2C_rc_linIdxs]
    edge2D_rc_linIdxs = np.ravel_multi_index((edge2D_rc[:, 0], edge2D_rc[:, 1]), imgD.shape)
    intensities_2D = np.ravel(imgD)[edge2D_rc_linIdxs]

    # original code for ravel multi index
    # edge1B_rc_linIdxs = sub2ind(np.shape(imgB), edge1B_rc[:, 0], edge1B_rc[:, 1])

    # Extend histograms and intensities to avoid indexing error
    a = 0.2
    hist_1A_ext = np.tile(hist_1A[-1, :], (int(np.round(a * np.shape(hist_1A)[0])), 1))
    hist_1A_large = np.append(hist_1A, hist_1A_ext, axis=0)
    hist_1B_ext = np.tile(hist_1B[-1, :], (int(np.round(a * np.shape(hist_1B)[0])), 1))
    hist_1B_large = np.append(hist_1B, hist_1B_ext, axis=0)
    hist_1C_ext = np.tile(hist_1C[-1, :], (int(np.round(a * np.shape(hist_1C)[0])), 1))
    hist_1C_large = np.append(hist_1C, hist_1C_ext, axis=0)
    hist_1D_ext = np.tile(hist_1D[-1, :], (int(np.round(a * np.shape(hist_1D)[0])), 1))
    hist_1D_large = np.append(hist_1D, hist_1D_ext, axis=0)

    hist_2A_ext = np.tile(hist_2A[-1, :], (int(np.round(a * np.shape(hist_2A)[0])), 1))
    hist_2A_large = np.append(hist_2A, hist_2A_ext, axis=0)
    hist_2B_ext = np.tile(hist_2B[-1, :], (int(np.round(a * np.shape(hist_2B)[0])), 1))
    hist_2B_large = np.append(hist_2B, hist_2B_ext, axis=0)
    hist_2C_ext = np.tile(hist_2C[-1, :], (int(np.round(a * np.shape(hist_2C)[0])), 1))
    hist_2C_large = np.append(hist_2C, hist_2C_ext, axis=0)
    hist_2D_ext = np.tile(hist_2D[-1, :], (int(np.round(a * np.shape(hist_2D)[0])), 1))
    hist_2D_large = np.append(hist_2D, hist_2D_ext, axis=0)

    intensities_1A_large = np.tile(intensities_1A, int(np.round(a * np.shape(intensities_1A)[0])))
    intensities_1B_large = np.tile(intensities_1B, int(np.round(a * np.shape(intensities_1B)[0])))
    intensities_1C_large = np.tile(intensities_1C, int(np.round(a * np.shape(intensities_1C)[0])))
    intensities_1D_large = np.tile(intensities_1D, int(np.round(a * np.shape(intensities_1D)[0])))
    intensities_2A_large = np.tile(intensities_2A, int(np.round(a * np.shape(intensities_2A)[0])))
    intensities_2B_large = np.tile(intensities_2B, int(np.round(a * np.shape(intensities_2B)[0])))
    intensities_2C_large = np.tile(intensities_2C, int(np.round(a * np.shape(intensities_2C)[0])))
    intensities_2D_large = np.tile(intensities_2D, int(np.round(a * np.shape(intensities_2D)[0])))

    # Fill histogram dict
    names_hist = ['hist_1A', 'hist_1B', 'hist_1C', 'hist_1D',
                  'hist_2A', 'hist_2B', 'hist_2C', 'hist_2D']
    histograms = [hist_1A_large, hist_1B_large, hist_1C_large, hist_1D_large,
                  hist_2A_large, hist_2B_large, hist_2C_large, hist_2D_large]

    h_dict = dict()
    for key, value in zip(names_hist, histograms):
        h_dict[str(key)] = value

    # Fill intensity dict
    names_int = ['1A', '1B', '1C', '1D', '2A', '2B', '2C', '2D']
    intensities = [intensities_1A_large, intensities_1B_large, intensities_1C_large, intensities_1D_large,
                   intensities_2A_large, intensities_2B_large, intensities_2C_large, intensities_2D_large]

    i_dict = dict()
    for key, value in zip(names_int, intensities):
        i_dict[str(key)] = value

    return h_dict, i_dict
