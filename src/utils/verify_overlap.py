import numpy as np
import matplotlib.pyplot as plt


def verify_non_overlap(quadrant_A, quadrant_B, quadrant_C, quadrant_D, tolerance):
    """
    Custom helper function to verify that there is no overlap between the quadrants
    after the initial alignment. If this is the case, the initial alignment is incorrect
    """

    assert len(np.unique(quadrant_A.mask)) == 2, "mask must be binary"
    assert len(np.unique(quadrant_B.mask)) == 2, "mask must be binary"
    assert len(np.unique(quadrant_C.mask)) == 2, "mask must be binary"
    assert len(np.unique(quadrant_D.mask)) == 2, "mask must be binary"

    fusion_AB = quadrant_A.mask + quadrant_B.mask
    fusion_BC = quadrant_B.mask + quadrant_C.mask
    fusion_CD = quadrant_C.mask + quadrant_D.mask
    fusion_DA = quadrant_D.mask + quadrant_A.mask
    im_pairs = ["A and B", "B and C", "C and D", "D and A"]

    for im, pair in zip([fusion_AB, fusion_BC, fusion_CD, fusion_DA], im_pairs):
        if len(np.unique(im)) > 2:

            # A few overlapping voxels is fine, but this must not be a large part
            # of the image as this may indicate improper initial alignment.
            max_val = np.unique(im)
            n_voxels = np.sum(im == max_val)

            if n_voxels > tolerance:

                plt.figure()
                plt.title(f"Found overlap for quadrants {pair}")
                plt.imshow(im, cmap="gray")
                plt.show()

                raise ValueError(f"Found overlapping masks for {pair}")

    return
