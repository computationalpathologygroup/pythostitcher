import numpy as np
import matplotlib.pyplot as plt
import multiresolutionimageinterface as mir
import pandas as pd

from scipy.spatial.distance import cdist


class LandmarkEvaluator:

    def __init__(self, parameters):

        self.n_fragments = parameters["n_fragments"]
        self.image_dir = parameters["sol_save_dir"].joinpath("highres")
        self.coord_dir = parameters["sol_save_dir"].joinpath("highres", "eval")
        self.output_res = parameters["output_res"]

        return

    def get_pairs(self):
        """
        Method to get all unique line pairs. We compute these based on the location
        of the center points.
        """

        # Get all lines
        self.all_lines = []
        for i in range(self.n_fragments):
            coords = np.load(
                str(self.coord_dir.joinpath(f"fragment{i + 1}_coordinates.npy")),
                allow_pickle=True
            ).item()

            self.all_lines.append(coords["a"])

            if self.n_fragments == 4:
                self.all_lines.append(coords["b"])

        # Match lines based on center. This is more robust against cases with varying number
        # of fragments
        self.center_points = np.array([np.mean(i, axis=0) for i in self.all_lines])

        # Get all unique line combinations
        self.all_pairs = []
        for line1_idx, cp in enumerate(self.center_points):

            # Get closest line based on center point
            distances = np.squeeze(cdist(cp[np.newaxis, :], self.center_points))
            line2_idx = np.argsort(distances)[1]

            pair = sorted([line1_idx, line2_idx])
            if not pair in self.all_pairs:
                self.all_pairs.append(pair)

        # Sanity check
        assert len(np.unique(self.all_pairs)) == len(self.all_lines), \
            f"could not find {int(len(self.all_lines) / 2)} unique line pairs"

        return

    def get_distances(self):
        """
        Method to compute the average distance between stitch lines.
        """

        self.all_line_distances = dict()

        for count, (line1_idx, line2_idx) in enumerate(self.all_pairs, 1):

            # Get lines and sort them so we match the right coordinates of both lines
            line1 = self.all_lines[line1_idx]
            is_hor = True if np.std(line1[:, 0]) > np.std(line1[:, 1]) else False
            line1 = sorted(line1, key=lambda x: x[0]) if is_hor else sorted(line1,
                                                                            key=lambda x: x[1])
            line1 = np.array(line1)

            line2 = self.all_lines[line2_idx]
            is_hor = True if np.std(line2[:, 0]) > np.std(line2[:, 1]) else False
            line2 = sorted(line2, key=lambda x: x[0]) if is_hor else sorted(line2,
                                                                            key=lambda x: x[1])
            line2 = np.array(line2)

            # Compute distances
            distances = [np.float(cdist(line1[i][np.newaxis, :], line2[i][np.newaxis, :])) \
                         for i in range(len(line1))]

            self.all_line_distances[f"stitch_line_{count}"] = distances

        return

    def scale_distances(self):
        """
        Method to scale the distances computed previously with the spacing.
        """

        # Load stitched image
        self.opener = mir.MultiResolutionImageReader()
        self.image_path = str(
            self.image_dir.joinpath(f"stitched_image_{self.output_res}_micron.tif")
        )
        self.image = self.opener.open(str(self.image_path))

        # Get scale factor from pixel spacing
        self.spacing = self.image.getSpacing()[0]

        # Scale keys by spacing so we get the distance in micron
        for key in self.all_line_distances.keys():
            self.all_line_distances[key] = [i*self.spacing for i in self.all_line_distances[key]]

        return

    def save_results(self):
        """
        Method to save the results from the residual registration error computation
        """

        # Save all distances between points and reference corresponding line
        self.df = pd.DataFrame()
        fragment_names = []
        fragment_values = []
        for key, value in self.all_line_distances.items():
            names = [key] * len(value)
            fragment_names.extend(names)
            fragment_values.extend(value)

        self.df["stitch_line"] = fragment_names
        self.df["dist_in_micron"] = fragment_values

        self.df.to_csv(str(self.coord_dir.joinpath("residual_error.csv")))

        return

    def sanity_check(self):
        """
        Method to perform a sanity check and plot all the lines on the image.
        """

        # Get image closest to 2000 pixels
        best_image_output_dims = 2000
        all_image_dims = [
            self.image.getLevelDimensions(i) for i in range(self.image.getNumberOfLevels())
        ]
        sanity_level = np.argmin([(i[0] - best_image_output_dims) ** 2 for i in all_image_dims])
        sanity_downsampling = self.image.getLevelDownsample(int(sanity_level))

        image_ds = self.image.getUCharPatch(
            startX=0,
            startY=0,
            width=int(all_image_dims[sanity_level][0]),
            height=int(all_image_dims[sanity_level][1]),
            level=int(sanity_level),
        )

        # Scale distance to pixels for easier interpretation
        sanity_all_dist = self.df["dist_in_micron"].tolist()
        sanity_all_dist = [i / (self.spacing * sanity_downsampling) for i in sanity_all_dist]
        sanity_avg_dist = np.mean(sanity_all_dist)

        # Plot image and all lines
        plt.figure()
        plt.title(f"avg dist: {sanity_avg_dist:.2f} pixels")
        plt.imshow(image_ds)

        for idx1, idx2 in self.all_pairs:
            line1 = self.all_lines[idx1] / sanity_downsampling
            line2 = self.all_lines[idx2] / sanity_downsampling
            plt.scatter(line1[:, 0], line1[:, 1], c="cornflowerblue")
            plt.scatter(line2[:, 0], line2[:, 1], c="darkblue")

        plt.savefig(str(self.coord_dir.joinpath("residual_error_figure.png")))
        plt.close()

        return


def evaluate_landmarks(parameters):
    """
    This function computes the residual registration error for each of the stitch lines.
    """

    eval = LandmarkEvaluator(parameters)
    eval.get_pairs()
    eval.get_distances()
    eval.scale_distances()
    eval.save_results()
    eval.sanity_check()

    return
