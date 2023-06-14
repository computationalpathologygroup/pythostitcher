import numpy as np
import itertools
import cv2
import copy
import math
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from scipy.spatial import distance

from .pairwise_alignment_utils import sort_counterclockwise


class Assembler:
    """
    Class for a case on which we perform global assembly.
    """

    def __init__(self, parameters):

        self.case_path = parameters["save_dir"].joinpath("configuration_detection")
        self.case_idx = self.case_path.name
        self.result_path = self.case_path.joinpath("global_assembly")
        if not self.result_path.is_dir():
            self.result_path.mkdir()
        self.score_type = parameters["alignment_score"]
        self.log = parameters["log"]

        return

    def check_case_eligibility(self):
        """
        Check data integrity of the directory. This simply runs a check whether
        all required files are available.
        """

        self.available_files = list(self.case_path.iterdir())
        self.mandatory_files = [
            "filtered_alignments.txt",
            "bg_color.txt",
            "fragment_list.txt",
            "fragment*.png",
        ]

        self.all_mandatory_files_present = all(
            [any([i.match(f"*{j}") for i in self.available_files]) for j in self.mandatory_files]
        )

        assert (
            self.all_mandatory_files_present
        ), f"> error: case {self.case_path} does not contain all required files"

        return

    def process_input_files(self):
        """
        Extract all data from the input .txt files
        """

        # Get relevant files
        bg_color_file = self.case_path.joinpath("bg_color.txt")
        fragment_list_file = self.case_path.joinpath("fragment_list.txt")
        alignments_file = self.case_path.joinpath("alignments.txt")
        filtered_alignments_file = self.case_path.joinpath("filtered_alignments.txt")
        stitch_edge_file = self.case_path.joinpath("stitch_edges.txt")

        # Process bg color file
        with open(bg_color_file, "r") as f:
            contents = f.readlines()[0]
            self.bg_color = [int(i) for i in contents.split(" ")]

        # Process fragment list file and load fragments
        with open(fragment_list_file, "r") as f:
            contents = f.readlines()
            self.fragment_list = [i.rstrip("\n") for i in contents]

        if self.score_type == "jigsawnet":

            # Process filtered alignments file
            with open(filtered_alignments_file, "r") as f:
                contents = f.readlines()

                n_combinations = int(len(contents) / 4)
                combinations_data = [contents[i * 4] for i in range(n_combinations)]
                combinations_data = [
                    i.replace("\t", " ").replace("\n", "") for i in combinations_data
                ]
                combinations_data = [i.rstrip(" 1") for i in combinations_data]

                self.fa_tforms = [contents[i * 4 + 1 : i * 4 + 4] for i in range(n_combinations)]
                self.fa_tforms = [[i.replace("\n", "") for i in tform] for tform in self.fa_tforms]
                self.fa_tforms = [" ".join(tform) for tform in self.fa_tforms]
                self.fa_tforms = [
                    np.array([float(i) for i in tform.split(" ")]) for tform in self.fa_tforms
                ]
                self.fa_tforms = [tform.reshape((3, 3)) for tform in self.fa_tforms]

                self.fa_combinations = [i.split(" ")[:2] for i in combinations_data]
                self.fa_combinations = [[int(i) for i in combo] for combo in self.fa_combinations]
                self.fa_scores = [i.split(" ")[2] for i in combinations_data]
                self.fa_scores = [float(i) for i in self.fa_scores]

        elif self.score_type == "pairwise_alignment":

            # Process alignments file
            with open(alignments_file, "r") as f:
                contents = f.readlines()
                contents = [i.split(" ") for i in contents if not "Node" in i]

                self.fa_combinations = [[int(i[0]), int(i[1])] for i in contents]
                self.fa_scores = [int(i[2]) for i in contents]
                self.fa_tforms = [
                    np.array(list(map(float, i[3:12]))).reshape(3, 3) for i in contents
                ]

        # Process stitch edge label file
        self.stitch_edge_dict = dict()
        with open(stitch_edge_file, "r") as f:
            contents = f.readlines()
            contents = [i.rstrip("\n") for i in contents]
            keys = [i.split(":")[0] for i in contents]
            values = [i.split(":")[1] for i in contents]
            for k, v in zip(keys, values):
                self.stitch_edge_dict[k] = v

        return

    def process_input_images(self):
        """
        Process all image related data
        """

        # Load fragments
        self.fragments = [cv2.imread(str(self.case_path.joinpath(i))) for i in self.fragment_list]
        self.fragments = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in self.fragments]

        # Compute masks and contours
        self.masks = [
            (np.all(i != self.bg_color, axis=2) * 255).astype("uint8") for i in self.fragments
        ]
        self.masks_cnts = [
            np.squeeze(
                max(
                    cv2.findContours(i, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0],
                    key=cv2.contourArea,
                )
            )
            for i in self.masks
        ]
        self.masks_cnts = [
            np.hstack([cnt[:, ::-1], np.ones((cnt.shape[0], 1))]) for cnt in self.masks_cnts
        ]

        return

    def get_feasible_configurations(self):
        """
        Find all potential configurations.
        """

        # Compute all possible configurations
        n_fragments = 4
        self.pair_combinations = [list(x) for x in set(tuple(x) for x in self.fa_combinations)]
        self.all_configurations = list(itertools.combinations(self.pair_combinations, n_fragments))
        self.valid_configurations = []

        # Filter out infeasible configurations
        for config in self.all_configurations:
            config_flat = np.array(config).ravel()

            # A configuration is only valid if each fragment is present twice
            is_valid_config = all([np.sum(config_flat == i) == 2 for i in range(n_fragments)])
            if is_valid_config:
                self.valid_configurations.append(list(config))

        return

    def get_solutions_per_configuration(self):
        """
        Function to look for all solutions for all possible configurations.
        """

        self.all_topn_idx_solutions = []

        # Get all combinations
        for config in self.valid_configurations:

            # Extract all pair combinations
            pair1, pair2, pair3, pair4 = config

            # Find indices of associated tforms
            pair1_all_idx = np.where([pair1 == i for i in self.fa_combinations])[0]
            pair2_all_idx = np.where([pair2 == i for i in self.fa_combinations])[0]
            pair3_all_idx = np.where([pair3 == i for i in self.fa_combinations])[0]
            pair4_all_idx = np.where([pair4 == i for i in self.fa_combinations])[0]

            # Find indices of n best solutions
            n_best = 4
            pair1_sorting = np.argsort([self.fa_scores[i] for i in pair1_all_idx])[-n_best:]
            pair1_topn_idx = [pair1_all_idx[i] for i in pair1_sorting]
            pair2_sorting = np.argsort([self.fa_scores[i] for i in pair2_all_idx])[-n_best:]
            pair2_topn_idx = [pair2_all_idx[i] for i in pair2_sorting]
            pair3_sorting = np.argsort([self.fa_scores[i] for i in pair3_all_idx])[-n_best:]
            pair3_topn_idx = [pair3_all_idx[i] for i in pair3_sorting]
            pair4_sorting = np.argsort([self.fa_scores[i] for i in pair4_all_idx])[-n_best:]
            pair4_topn_idx = [pair4_all_idx[i] for i in pair4_sorting]

            # Get all possible combinations of taking 1 item from each list
            topn_idx_solutions = list(itertools.product(range(n_best), repeat=4))
            self.topn_idx_solutions = [
                [pair1_topn_idx[i1], pair2_topn_idx[i2], pair3_topn_idx[i3], pair4_topn_idx[i4]]
                for i1, i2, i3, i4 in topn_idx_solutions
            ]

            self.all_topn_idx_solutions.append(self.topn_idx_solutions)

        self.all_topn_idx_solutions = [np.array(i) for i in self.all_topn_idx_solutions]
        self.all_topn_idx_solutions = np.vstack(self.all_topn_idx_solutions)

        # Sort all solutions based on score
        self.all_topn_idx_scores = [
            np.sum([self.fa_scores[i] for i in s]) for s in self.all_topn_idx_solutions
        ]
        self.all_topn_idx_sort_key = np.argsort(self.all_topn_idx_scores)[::-1]

        self.all_topn_sorted_solutions = np.take_along_axis(
            self.all_topn_idx_solutions, self.all_topn_idx_sort_key[:, np.newaxis], axis=0
        )

        return

    def evaluate_solutions(self):
        """
        Function to scout feasibility of solutions.
        """

        # Iterate over all potential solutions and obtain the feasible ones.
        for count, solution in enumerate(self.all_topn_sorted_solutions):

            # Extract combinations and associated tforms
            combinations = [self.fa_combinations[i] for i in solution]
            tforms = [self.fa_tforms[i] for i in solution]
            self.sol_idx = count
            self.sol_score = np.sum([self.fa_scores[i] for i in solution])

            # Set initial feasibility state. Can be set to True when some assembly
            # constraints are met. Positive state will call final image assembly function.
            self.feasible_assembly = False

            # Perform initial reconstruction and check some assembly constraints. When
            # constraints are met, continue full image assembly.
            self.get_tform_per_fragment(combinations, tforms)
            self.verify_feasibility()

            # If feasible, this is set to True in the verify feasibility method
            if self.feasible_assembly:
                self.compute_assembly_score()
                self.perform_assembly()

        return

    def get_tform_per_fragment(self, combinations, tforms):
        """
        Function to obtain the transformation matrix per fragment.
        """

        # Get the two image indices that are related to the reference image
        self.frag_idx_ref = 0
        self.init_combinations = np.where([i[0] == self.frag_idx_ref for i in combinations])[0]
        self.frag_idx1 = combinations[self.init_combinations[0]][1]
        self.frag_idx2 = combinations[self.init_combinations[1]][1]
        self.frag_idx1_2 = list(np.unique([combinations[i] for i in self.init_combinations]))
        self.tform_idx1 = self.init_combinations[0]
        self.tform_idx2 = self.init_combinations[1]

        # Get the image index that is not directly related to the reference image
        self.all_fragment_idx = list(range(len(self.fragments)))
        self.frag_idx3 = (set(self.all_fragment_idx) - set(self.frag_idx1_2)).pop()

        # Sanity check to see if all tforms are used
        check_all_fragment_idx = list(self.frag_idx1_2) + [self.frag_idx3]
        assert len(np.unique(check_all_fragment_idx)) == len(
            tforms
        ), "error figuring out fragment configuration"

        # Find the counterpart of the final piece. We might need to invert this tform
        # depending on its configuration.
        if [self.frag_idx1, self.frag_idx3] in combinations:
            self.tform_idx3 = combinations.index([self.frag_idx1, self.frag_idx3])
            self.inv_tform3 = False
        elif [self.frag_idx3, self.frag_idx1] in combinations:
            self.tform_idx3 = combinations.index([self.frag_idx3, self.frag_idx1])
            self.inv_tform3 = True
        else:
            raise ValueError("Could not find fourth piece")

        # Obtain final tform for closing the loop later
        if [self.frag_idx2, self.frag_idx3] in combinations:
            self.tform_idx_ = combinations.index([self.frag_idx2, self.frag_idx3])
            self.inv_tform_ = False
        elif [self.frag_idx3, self.frag_idx2] in combinations:
            self.tform_idx_ = combinations.index([self.frag_idx3, self.frag_idx2])
            self.inv_tform_ = True
        else:
            raise ValueError("Wrong loop configuration")

        # Obtain reference transformation for each dst image
        self.tform_dict = dict()
        self.tform_dict[str(self.frag_idx1)] = tforms[self.tform_idx1]
        self.tform_dict[str(self.frag_idx2)] = tforms[self.tform_idx2]
        if self.inv_tform3:
            self.tform_dict[str(self.frag_idx3)] = np.linalg.inv(tforms[self.tform_idx3])
        else:
            self.tform_dict[str(self.frag_idx3)] = tforms[self.tform_idx3]
        if self.inv_tform_:
            tforms[self.tform_idx_] = np.linalg.inv(tforms[self.tform_idx_])

        ### Convert the point transformation matrices to image transformation matices. ###

        # Arbitrary choice to guarantee enough space around the reference image
        self.output_size = (6000, 6000)
        self.t_center = 2000

        # Compute point transform for each image
        self.offset_tform_src = np.array([[1, 0, self.t_center], [0, 1, self.t_center], [0, 0, 1]])

        self.point_tform_ref_to_1 = self.offset_tform_src @ self.tform_dict[str(self.frag_idx1)]
        self.point_tform_ref_to_2 = self.offset_tform_src @ self.tform_dict[str(self.frag_idx2)]
        self.point_tform_ref_to_1_to_3 = (
            self.point_tform_ref_to_1 @ self.tform_dict[str(self.frag_idx3)]
        )
        self.point_tform_ref_to_2_to_3 = self.point_tform_ref_to_2 @ tforms[self.tform_idx_]

        # Required conversion between row/col to opencv x/y convention
        self.img_dst_tform1 = np.float32(
            [
                [
                    self.point_tform_ref_to_1[0, 0],
                    self.point_tform_ref_to_1[1, 0],
                    self.point_tform_ref_to_1[1, 2],
                ],
                [
                    self.point_tform_ref_to_1[0, 1],
                    self.point_tform_ref_to_1[1, 1],
                    self.point_tform_ref_to_1[0, 2],
                ],
            ]
        )
        self.img_dst_tform2 = np.float32(
            [
                [
                    self.point_tform_ref_to_2[0, 0],
                    self.point_tform_ref_to_2[1, 0],
                    self.point_tform_ref_to_2[1, 2],
                ],
                [
                    self.point_tform_ref_to_2[0, 1],
                    self.point_tform_ref_to_2[1, 1],
                    self.point_tform_ref_to_2[0, 2],
                ],
            ]
        )
        self.img_dst_tform3a = np.float32(
            [
                [
                    self.point_tform_ref_to_1_to_3[0, 0],
                    self.point_tform_ref_to_1_to_3[1, 0],
                    self.point_tform_ref_to_1_to_3[1, 2],
                ],
                [
                    self.point_tform_ref_to_1_to_3[0, 1],
                    self.point_tform_ref_to_1_to_3[1, 1],
                    self.point_tform_ref_to_1_to_3[0, 2],
                ],
            ]
        )
        self.img_dst_tform3b = np.float32(
            [
                [
                    self.point_tform_ref_to_2_to_3[0, 0],
                    self.point_tform_ref_to_2_to_3[1, 0],
                    self.point_tform_ref_to_2_to_3[1, 2],
                ],
                [
                    self.point_tform_ref_to_2_to_3[0, 1],
                    self.point_tform_ref_to_2_to_3[1, 1],
                    self.point_tform_ref_to_2_to_3[0, 2],
                ],
            ]
        )

        # Required to prevent math domain errors later due to very tiny disturbances
        self.img_dst_tform1 = np.round(self.img_dst_tform1, 4)
        self.img_dst_tform2 = np.round(self.img_dst_tform2, 4)
        self.img_dst_tform3a = np.round(self.img_dst_tform3a, 4)
        self.img_dst_tform3b = np.round(self.img_dst_tform3b, 4)

        return

    def verify_feasibility(self):
        """
        Function to check several constraints of the proposed reassembly. These constraints
        are computed over point coordinates rather than over images for a significant
        speedup. Once these initial constraints are met, the actual image assembly will
        take place.
        """

        ###################################################
        # CONSTRAINT no. 1 - full puzzle configuration
        ###################################################

        # Obtain number of 90 deg rotation steps the image has undergone. We use this to
        # determine the new stitch edge location.
        img1_rot = math.degrees(math.acos(self.img_dst_tform1[0, 0]))
        img2_rot = math.degrees(math.acos(self.img_dst_tform2[0, 0]))
        img3a_rot = math.degrees(math.acos(self.img_dst_tform3a[0, 0]))
        img3b_rot = math.degrees(math.acos(self.img_dst_tform3b[0, 0]))

        # Compensate for potential flipping.
        img1_rot = (360 - img1_rot) if self.img_dst_tform1[0, 1] < 0 else img1_rot
        img2_rot = (360 - img2_rot) if self.img_dst_tform2[0, 1] < 0 else img2_rot
        img3a_rot = (360 - img3a_rot) if self.img_dst_tform3a[0, 1] < 0 else img3a_rot
        img3b_rot = (360 - img3b_rot) if self.img_dst_tform3b[0, 1] < 0 else img3b_rot

        img1_rot_steps = int(np.round(img1_rot / 90))
        img2_rot_steps = int(np.round(img2_rot / 90))
        img3a_rot_steps = int(np.round(img3a_rot / 90))
        img3b_rot_steps = int(np.round(img3b_rot / 90))

        # Obtain old stitch edge labels per fragment
        self.ref_img_old_label = self.stitch_edge_dict[
            f"fragment{self.frag_idx_ref+1}.png"
        ]
        self.img1_old_label = self.stitch_edge_dict[
            f"fragment{self.frag_idx1+1}.png"
        ]
        self.img2_old_label = self.stitch_edge_dict[
            f"fragment{self.frag_idx2+1}.png"
        ]
        self.img3a_old_label = self.stitch_edge_dict[
            f"fragment{self.frag_idx3+1}.png"
        ]
        self.img3b_old_label = self.stitch_edge_dict[
            f"fragment{self.frag_idx3+1}.png"
        ]

        # Obtain new tformed stitch edge labels per fragment
        self.all_stitch_labels = ["UL", "UR", "LR", "LL"]
        self.ref_img_new_label = copy.copy(self.ref_img_old_label)
        self.img1_new_label = self.all_stitch_labels[
            self.all_stitch_labels.index(self.img1_old_label) - img1_rot_steps
        ]
        self.img2_new_label = self.all_stitch_labels[
            self.all_stitch_labels.index(self.img2_old_label) - img2_rot_steps
        ]
        self.img3a_new_label = self.all_stitch_labels[
            self.all_stitch_labels.index(self.img3a_old_label) - img3a_rot_steps
        ]
        self.img3b_new_label = self.all_stitch_labels[
            self.all_stitch_labels.index(self.img3b_old_label) - img3b_rot_steps
        ]
        new_labels = set(
            [self.ref_img_new_label, self.img1_new_label, self.img2_new_label, self.img3a_new_label]
        )

        # All labels must occur exactly once AND label of final piece must be consistent
        constraint_1 = (len(new_labels) == 4) and (self.img3a_new_label == self.img3b_new_label)

        #########################################################
        # CONSTRAINT no. 2 - maximum rotation/translation error
        #########################################################

        # Final tform to close the loop
        self.tform_loop = (
            np.linalg.inv(self.point_tform_ref_to_2_to_3) @ self.point_tform_ref_to_1_to_3
        )
        self.tform_loop_v2 = self.point_tform_ref_to_1_to_3 @ np.linalg.inv(
            self.point_tform_ref_to_2_to_3
        )

        # Translation and rotation error. This metric is not really consistent and has some
        # overlap with constraint 3. Might as well exclude this and just focus on constraint 3.
        t_err = np.sqrt(self.tform_loop[0, 2] ** 2 + self.tform_loop[1, 2] ** 2)
        r_err = np.abs(math.degrees(math.acos(np.round(self.tform_loop[0, 0], 4))))

        t_thres = 500
        r_thres = 20

        ###################################################
        # CONSTRAINT no. 3 - minimum and maximum overlap
        ###################################################

        # Get transformed contours
        self.cnt_ref = self.offset_tform_src @ self.masks_cnts[self.frag_idx_ref].T
        self.cnt_ref_2d = np.array([self.cnt_ref[1, :], self.cnt_ref[0, :]]).T

        self.cnt_idx1 = (
            np.vstack([self.point_tform_ref_to_1, np.array([0, 0, 1])])
            @ self.masks_cnts[self.frag_idx1].T
        )
        self.cnt_idx1_2d = np.array([self.cnt_idx1[1, :], self.cnt_idx1[0, :]]).T

        self.cnt_idx2 = (
            np.vstack([self.point_tform_ref_to_2, np.array([0, 0, 1])])
            @ self.masks_cnts[self.frag_idx2].T
        )
        self.cnt_idx2_2d = np.array([self.cnt_idx2[1, :], self.cnt_idx2[0, :]]).T

        self.cnt_idx3a = (
            np.vstack([self.point_tform_ref_to_1_to_3, np.array([0, 0, 1])])
            @ self.masks_cnts[self.frag_idx3].T
        )
        self.cnt_idx3a_2d = np.array([self.cnt_idx3a[1, :], self.cnt_idx3a[0, :]]).T

        self.cnt_idx3b = (
            np.vstack([self.point_tform_ref_to_2_to_3, np.array([0, 0, 1])])
            @ self.masks_cnts[self.frag_idx3].T
        )
        self.cnt_idx3b_2d = np.array([self.cnt_idx3b[1, :], self.cnt_idx3b[0, :]]).T

        # Get averaged contour between 3a and 3b
        self.cnt_idx3_2d = (self.cnt_idx3a_2d + self.cnt_idx3b_2d) / 2

        # Convert to shapely polygon format
        self.cnt_ref_pol = Polygon(self.cnt_ref_2d)
        self.cnt_idx1_pol = Polygon(self.cnt_idx1_2d)
        self.cnt_idx2_pol = Polygon(self.cnt_idx2_2d)
        self.cnt_idx3a_pol = Polygon(self.cnt_idx3a_2d)
        self.cnt_idx3b_pol = Polygon(self.cnt_idx3b_2d)
        self.cnt_idx3_pol = Polygon(self.cnt_idx3_2d)

        # Compute overlap between relevant fragments
        frag12_overlap = (2 * self.cnt_idx1_pol.intersection(self.cnt_idx2_pol).area) / (
            self.cnt_idx1_pol.area + self.cnt_idx2_pol.area
        )
        frag13_overlap = (2 * self.cnt_ref_pol.intersection(self.cnt_idx3a_pol).area) / (
            self.cnt_ref_pol.area + self.cnt_idx3_pol.area
        )
        frag33_overlap = (2 * self.cnt_idx3a_pol.intersection(self.cnt_idx3b_pol).area) / (
            self.cnt_idx3a_pol.area + self.cnt_idx3b_pol.area
        )

        # Fragment 1~2 and 1~3 cannot have overlap.
        overlap_thresh_max = 0.2
        max_overlap_constraint = all(
            [i < overlap_thresh_max for i in [frag12_overlap, frag13_overlap]]
        )

        # Fragment 3a and 3b must have overlap
        overlap_thresh_min = 0.5
        min_overlap_constraint = frag33_overlap > overlap_thresh_min
        constraint_3 = min_overlap_constraint and max_overlap_constraint

        # Final verdict. If true, perform full assembly. If false, move to next solution.
        self.feasible_assembly = all([constraint_1, constraint_3])

        ### SANITY CHECK ###
        # if self.feasible_assembly:
        #     plt.figure(figsize=(8, 8))
        #     all_cnts = [self.cnt_ref_2d, self.cnt_idx1_2d, self.cnt_idx2_2d, self.cnt_idx3a_2d,
        #                 self.cnt_idx3b_2d]
        #     colours = ["b", "r", "r", "g", "g"]
        #     for col, cnt in zip(colours, all_cnts):
        #         plt.title(
        #             f"123 overlap: {frag12_overlap:.2f} & {frag13_overlap:.2f}\n33 overlap: {frag33_overlap:.2f}\nT_err: {t_err:.2f} & R_err: {r_err:.2f}")
        #         plt.plot(cnt[:, 0], cnt[:, 1], c=col)
        #     # plt.plot(test[:, 0], test[:, 1], c="g", linewidth=4)
        #     plt.gca().invert_yaxis()
        #     plt.show()

        return

    def compute_assembly_score(self):
        """
        Function to compute the fit of the assembly. We use this to later rank the
        different feasible assembly solutions.
        """

        # Get horizontal and vertical stitch line per fragment
        all_contours = [self.cnt_ref_2d, self.cnt_idx1_2d, self.cnt_idx2_2d, self.cnt_idx3_2d]
        self.all_labels = [
            self.ref_img_new_label,
            self.img1_new_label,
            self.img2_new_label,
            self.img3a_new_label,
        ]
        all_names = ["ref", "1", "2", "3"]
        self.stitch_line_dict = dict()

        for cnt, label, name in zip(all_contours, self.all_labels, all_names):

            # Get bbox around the contour
            cnt = cnt.astype("int")
            cnt_bbox = cv2.minAreaRect(cnt)
            cnt_bbox_corners = cv2.boxPoints(cnt_bbox)

            # Compute distance from bbox corners to the contour
            distances = distance.cdist(cnt_bbox_corners, cnt)
            indices = np.argmin(distances, axis=1)
            cnt_corners = np.array([list(cnt[i, :]) for i in indices])
            cnt_corners = sort_counterclockwise(cnt_corners)
            cnt_corners_loop = np.vstack([cnt_corners, cnt_corners])
            ul_cnt_corner_idx = np.argmin(np.sum(cnt_corners, axis=1) ** 2)

            # Get both the horizontal and vertical stitch line of each fragment
            cnt_fragments = []
            for i in range(2):
                cnt_ccw = sort_counterclockwise(cnt)

                # Get starting point and end point of a stitch edge
                label_idx = self.all_stitch_labels.index(label)
                start_corner = cnt_corners_loop[ul_cnt_corner_idx + label_idx - 1 + i]
                end_corner = cnt_corners_loop[ul_cnt_corner_idx + label_idx + i]

                start_idx = np.argmax((cnt_ccw == start_corner).all(axis=1))
                end_idx = np.argmax((cnt_ccw == end_corner).all(axis=1))

                # Get the full stitch line
                if end_idx > start_idx:
                    cnt_fragment = cnt_ccw[start_idx:end_idx]
                else:
                    cnt_fragment = np.vstack([cnt_ccw[start_idx:], cnt_ccw[:end_idx]])

                cnt_fragments.append(cnt_fragment)

            # Extract which line is the horizontal and which is the vertical
            hline_idx = np.argmax([np.std(i[:, 0]) for i in cnt_fragments])
            vline_idx = np.argmax([np.std(i[:, 1]) for i in cnt_fragments])

            # Resample horizontal line to 100 points for computational efficiency
            hline_x = cnt_fragments[hline_idx][:, 0]
            hline_y = cnt_fragments[hline_idx][:, 1]
            sample_idx = np.linspace(0, len(hline_x) - 1, 100).astype("int")
            hline_new_x = [hline_x[i] for i in sample_idx]
            hline_new_y = [hline_y[i] for i in sample_idx]
            hline_sampled = np.vstack([hline_new_x, hline_new_y]).T

            # Resample vertical line to 100 points for computational efficiency
            vline_x = cnt_fragments[vline_idx][:, 0]
            vline_y = cnt_fragments[vline_idx][:, 1]
            sample_idx = np.linspace(0, len(vline_y) - 1, 100).astype("int")
            vline_new_x = [vline_x[i] for i in sample_idx]
            vline_new_y = [vline_y[i] for i in sample_idx]
            vline_sampled = np.vstack([vline_new_x, vline_new_y]).T

            # Save both lines for each fragment
            self.stitch_line_dict[name] = [hline_sampled, vline_sampled]

        # Compute assembly score
        locations = ["upper", "lower", "left", "right"]
        self.mse_score = np.sum([self.compute_mse_stitch_line(location=l) for l in locations])

        # Write results for each solution to text file
        location_solution = self.case_path.joinpath("location_solution.txt")
        label2opposite = {"UR": "LL", "UL": "LR", "LR": "UL", "LL": "UR"}

        if location_solution.exists():
            with open(location_solution, "r") as f:
                prev = f.readlines()
            with open(location_solution, "w") as f:
                for line in prev:
                    f.write(line)
                f.write(
                    f"\nmse:{int(self.mse_score)},"
                    f"fragment{self.frag_idx_ref + 1}.png:{label2opposite[self.ref_img_new_label]},"
                    f"fragment{self.frag_idx1 + 1}.png:{label2opposite[self.img1_new_label]},"
                    f"fragment{self.frag_idx2 + 1}.png:{label2opposite[self.img2_new_label]},"
                    f"fragment{self.frag_idx3 + 1}.png:{label2opposite[self.img3a_new_label]}"
                )
        else:
            with open(location_solution, "w") as f:
                f.write(
                    f"mse:{int(self.mse_score)},"
                    f"fragment{self.frag_idx_ref + 1}.png:{label2opposite[self.ref_img_new_label]},"
                    f"fragment{self.frag_idx1 + 1}.png:{label2opposite[self.img1_new_label]},"
                    f"fragment{self.frag_idx2 + 1}.png:{label2opposite[self.img2_new_label]},"
                    f"fragment{self.frag_idx3 + 1}.png:{label2opposite[self.img3a_new_label]}"
                )

        return

    def compute_mse_stitch_line(self, location):
        """
        Helper function to compute the mean squared error between two stitch lines. The
        function takes into account the location where the final fragment is placed
        and average the metric for these locations.
        """

        # Get the labels, given the location. Note that the labels are related to the
        # location of the centerpoint of the fragment. Hence, for the stitch edge of the
        # upper two fragments, we need to compare the two fragments with the centerpoint
        # at lower left and lower right.
        loc2label = {
            "upper": ["LL", "LR"],
            "right": ["LL", "UL"],
            "lower": ["UR", "UL"],
            "left": ["UR", "LR"],
        }
        label2opposite = {"UR": "LL", "UL": "LR", "LR": "UL", "LL": "UR"}
        labels = loc2label[str(location)]

        # Determine whether we need horizontal or vertical line pairs
        hv_idx = (location in ["upper", "lower"]) * 1

        # Get both lines and compute mse
        line1 = list(self.stitch_line_dict.values())[self.all_labels.index(labels[0])][hv_idx]
        line2 = list(self.stitch_line_dict.values())[self.all_labels.index(labels[1])][hv_idx]
        mse12 = int(np.mean(np.min(distance.cdist(line1, line2), axis=0) ** 2))
        mse21 = int(np.mean(np.min(distance.cdist(line2, line1), axis=0) ** 2))
        mse = np.mean([mse12, mse21])

        ### SANITY CHECK ###
        # plt.figure(figsize=(6, 6))
        # plt.title(f"Loc: {location}, MSE_avg: {int(mse)}")
        # plt.plot(line1[:, 0], line1[:, 1], c="r")
        # plt.plot(line2[:, 0], line2[:, 1], c="b")
        # plt.legend([label2opposite[labels[0]], label2opposite[labels[1]]])
        # plt.axis('square')
        # plt.gca().invert_yaxis()
        # plt.show()

        return mse

    def perform_assembly(self):
        """
        Perform full image assembly.
        """

        # Convert background colour of all images to black
        black_bg = [0, 0, 0]
        if self.bg_color != black_bg:
            for im in self.fragments:
                im[np.where((im == self.bg_color).all(axis=2))] = [0, 0, 0]

        # Save and label images in dict
        images = dict()
        images[str(self.frag_idx_ref)] = self.fragments[self.frag_idx_ref]
        images[str(self.frag_idx1)] = self.fragments[self.frag_idx1]
        images[str(self.frag_idx2)] = self.fragments[self.frag_idx2]
        images[str(self.frag_idx3)] = self.fragments[self.frag_idx3]

        # Get all transformed images and masks
        tformed_images = dict()
        tformed_images[str(self.frag_idx_ref)] = cv2.warpAffine(
            images[str(self.frag_idx_ref)],
            self.offset_tform_src[:2, :].astype("float32"),
            self.output_size,
        )
        tformed_images[str(self.frag_idx1)] = cv2.warpAffine(
            images[str(self.frag_idx1)], self.img_dst_tform1, self.output_size
        )
        tformed_images[str(self.frag_idx2)] = cv2.warpAffine(
            images[str(self.frag_idx2)], self.img_dst_tform2, self.output_size
        )
        tformed_images[str(self.frag_idx3)] = cv2.warpAffine(
            images[str(self.frag_idx3)], self.img_dst_tform3a, self.output_size
        )
        tformed_images["_"] = cv2.warpAffine(
            images[str(self.frag_idx3)], self.img_dst_tform3b, self.output_size
        )

        ### SANITY CHECK ###
        # plt.figure(figsize=(6, 12))
        #
        # plt.subplot(421)
        # plt.title(self.img1_old_label)
        # plt.imshow(images[str(self.frag_idx1)])
        # plt.axis("off")
        # plt.subplot(422)
        # plt.title(self.img1_new_label)
        # plt.imshow(tformed_images[str(self.frag_idx1)])
        # plt.axis("off")
        #
        # plt.subplot(423)
        # plt.title(self.img2_old_label)
        # plt.imshow(images[str(self.frag_idx2)])
        # plt.axis("off")
        # plt.subplot(424)
        # plt.title(self.img2_new_label)
        # plt.imshow(tformed_images[str(self.frag_idx2)])
        # plt.axis("off")
        #
        # plt.subplot(425)
        # plt.title(self.img3a_old_label)
        # plt.imshow(images[str(self.frag_idx3)])
        # plt.axis("off")
        # plt.subplot(426)
        # plt.title(self.img3a_new_label)
        # plt.imshow(tformed_images[str(self.frag_idx3)])
        # plt.axis("off")
        #
        # plt.subplot(427)
        # plt.title(self.img3b_old_label)
        # plt.imshow(images[str(self.frag_idx3)])
        # plt.axis("off")
        # plt.subplot(428)
        # plt.title(self.img3b_new_label)
        # plt.imshow(tformed_images["_"])
        # plt.axis("off")
        #
        # plt.show()

        # Rough assembly by summing images. Doesn't need to be fancy
        assembled_image = np.sum(list(tformed_images.values())[:-1], axis=0)
        assembled_image = np.clip(assembled_image, 0, 255).astype("uint8")

        # Cropping for visualization purposes
        assembled_mask = np.all(assembled_image != [0, 0, 0], axis=2) * 1
        r, c = np.nonzero(assembled_mask)
        min_x = np.max([0, np.min(r) - 200])
        max_x = np.min([np.max(r) + 200, assembled_mask.shape[0]])
        min_y = np.max([0, np.min(c) - 200])
        max_y = np.min([np.max(c) + 200, assembled_mask.shape[1]])
        assembled_image_crop = assembled_image[min_x:max_x, min_y:max_y, :]

        # Plot result
        plt.figure(figsize=(8, 8))
        plt.title(f"JSN score = {self.sol_score:.2f}, MSE score = {int(self.mse_score)}")
        plt.imshow(assembled_image_crop)
        plt.axis("off")
        plt.savefig(self.result_path.joinpath(f"sol_{self.sol_idx}.png"))
        plt.close()

        ### SANITY CHECK ###
        # plt.figure()
        # plt.title(f"MSE: {self.mse_score:.2f} and T_err: {t_err:.2f}")
        # plt.plot(self.cnt_ref_2d[:, 0], self.cnt_ref_2d[:, 1], c="r")
        # plt.plot(self.cnt_idx1_2d[:, 0], self.cnt_idx1_2d[:, 1], c=[0, 0, 1])
        # plt.plot(self.cnt_idx2_2d[:, 0], self.cnt_idx2_2d[:, 1], c=[0, 0, 0.75])
        # plt.plot(self.cnt_idx3a_2d[:, 0], self.cnt_idx3a_2d[:, 1], c=[0, 1, 0])
        # plt.plot(self.cnt_idx3b_2d[:, 0], self.cnt_idx3b_2d[:, 1], c=[0, 0.75, 0])
        # plt.legend(["ref", "1", "2", "3a", "3b"])
        # plt.gca().invert_yaxis()
        # plt.savefig(self.result_path.joinpath(f"sol_{self.sol_idx}_debug.png"))
        # plt.close()

        print("solution found")
        self.log.log(45, "   -> found solution!")

        return
