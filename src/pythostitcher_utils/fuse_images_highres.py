import copy
import numpy as np
import cv2
import itertools


def is_valid_contour(cnt):
    """
    Function to check whether a given contour is valid. We define invalid contours as
    contours which are very small and which look more like an artefact rather than an
    actual contour that helps in gradient blending. We try to filter these invalid
    contours based on 3 criteria:
        - small total contour length
        - small domain of rows/cols
        - small total contour area

    If any of these criteria is met, the contour is deemed invalid and the contour
    will not be used for the gradient blending. Most cutoff values were chosen
    empirically.
    """

    # Prepare contour
    cnt = np.squeeze(cnt)

    # Criterium contour length
    if len(cnt) < 50:
        return False

    # Criterium domain
    xcoords = cnt[:, 0]
    ycoords = cnt[:, 1]

    if (len(np.unique(xcoords)) < 20) or (len(np.unique(ycoords)) < 20):
        return False

    # Criterium contour area
    area = cv2.contourArea(cnt)
    if area < 500:
        return False

    return True


def get_gradients(bbox, q1_mask, overlap, direction, pad):
    """
    Custom function to obtain the gradient based on the direction. This function
    will also apply the mask to the gradient.

    Input:
        - Bounding box of the overlap
        - Mask of fragment 1
        - Overlapping area
        - Direction of overlap

    Output:
        - Masked gradient of overlap
        - Inverse masked gradient of overlap
    """

    # Get some bbox values
    bbox_center = bbox[0]
    angle = bbox[2]

    # Preallocate gradient field
    gradient_2d = np.zeros_like(q1_mask)
    gradient_2d = np.pad(gradient_2d, [[pad, pad], [pad, pad]]).astype("float")

    # OpenCV provides angles in range [0, 90]. We rescale these values to range [-45, 45]
    # for our use case. One extra factor to take into account is that we have to swap
    # width and height for angles in original openCV range [45, 90].
    if bbox[2] < 45:
        angle = -angle
        width = int(bbox[1][0])
        height = int(bbox[1][1])
    elif bbox[2] >= 45:
        angle = 90 - angle
        width = int(bbox[1][1])
        height = int(bbox[1][0])

    # Get slicing locations
    xmin = int(bbox_center[0] - 0.5 * width)
    xmax = xmin + width
    ymin = int(bbox_center[1] - 0.5 * height)
    ymax = ymin + height

    # Create 2d gradient
    if direction == "horizontal":
        gradient_1d = np.linspace(1, 0, width)
        gradient_2d_fill = np.tile(gradient_1d, (height, 1))
    elif direction == "vertical":
        gradient_1d = np.linspace(1, 0, height)
        gradient_2d_fill = np.tile(gradient_1d, (width, 1))
        gradient_2d_fill = np.transpose(gradient_2d_fill)

    # Insert gradient in image and rotate it along its primary axis
    gradient_2d[ymin:ymax, xmin:xmax] = gradient_2d_fill
    rot_mat = cv2.getRotationMatrix2D(center=bbox_center, angle=angle, scale=1)
    gradient_2d = cv2.warpAffine(gradient_2d, rot_mat, dsize=gradient_2d.shape[::-1])
    gradient_2d = gradient_2d[pad:-pad, pad:-pad]

    # Apply overlap mask to gradient
    masked_gradient = gradient_2d * overlap

    # Get reverse gradient
    masked_gradient_rev = (1 - masked_gradient) * (masked_gradient > 0)

    return masked_gradient, masked_gradient_rev


def fuse_images_highres(images, masks):
    """
    Custom function to merge overlapping fragments into a visually appealing combined
    image using alpha blending. This is accomplished by the following steps:
    1. Compute areas of overlap between different fragments
    2. Compute bounding box around the overlap
    3. Compute alpha gradient field over this bounding box
    4. Mask the gradient field with the area of overlap
    5. Apply the masked gradient field to original image intensity
    6. Sum all resulting (non)overlapping images to create the final image

    Inputs
        - Final colour image of all fragments

    Output
        - Merged output image
    """

    # Get plausible overlapping fragments
    names = list(images.keys())
    combinations = itertools.combinations(names, 2)

    hor_combinations = [
		["ul", "ur"],
		["ul", "lr"],
		["ll", "ur"],
		["ll", "lr"],
		["left", "right"]
	]
    ver_combinations = [
		["ul", "ll"],
		["ul", "lr"],
		["ur", "ll"],
		["ur", "lr"],
		["top", "bottom"]
	]

    # Create some lists for iterating
    total_mask = np.sum(list(masks.values()), axis=0).astype("uint8")
    all_contours = []
    is_overlap_list = []
    overlapping_fragments = []

    # Create some dicts for saving results
    gradients = dict()
    overlaps = dict()
    nonoverlap = dict()

    # Some values for postprocessing
    patch_size_mean = 25

    # Loop over possible combinations of overlapping fragments
    for combi in combinations:

        # Ensure right direction in gradient
        all_combinations = hor_combinations + ver_combinations
        if list(combi) in all_combinations:
            q1_name, q2_name = combi
        elif list(combi[::-1]) in all_combinations:
            q2_name, q1_name = combi

        # Check if overlap is between horizontally or vertically aligned fragments
        is_horizontal = [q1_name, q2_name] in hor_combinations
        is_vertical = [q1_name, q2_name] in ver_combinations

        # Get fragments and masks
        q1_image = images[q1_name]
        q2_image = images[q2_name]
        q1_mask = masks[q1_name]
        q2_mask = masks[q2_name]

        # Check if there is any overlap
        overlap = np.squeeze(((q1_mask + q2_mask) == 2) * 1).astype("uint8")
        is_overlap = np.sum(overlap) > 0
        is_overlap_list.append(is_overlap)
        overlaps[(q1_name, q1_name)] = overlap

        # In case of overlap, apply alpha blending
        if is_overlap:

            # Save index of overlapping fragments
            overlapping_fragments.append([q1_name, q2_name])

            # Compute non overlapping part of fragments
            only_q1 = q1_image * (total_mask == 1)[:, :, np.newaxis]
            only_q2 = q2_image * (total_mask == 1)[:, :, np.newaxis]
            nonoverlap[q1_name] = only_q1
            nonoverlap[q2_name] = only_q2

            # When (nearly) entire image is overlap, we blend by using the average of
            # both images. We implement a small tolerance value (1%) to still apply 50/50
            # blending in case of a few stray voxels.
            eps = int((np.shape(overlap)[0] * np.shape(overlap)[1]) / 100)
            approx_max_overlap = np.shape(overlap)[0] * np.shape(overlap)[1] - eps
            if np.sum(overlap) > approx_max_overlap:
                gradients[(q1_name, q2_name)] = np.full(overlap.shape, 0.5)
                gradients[(q2_name, q1_name)] = np.full(overlap.shape, 0.5)
                continue

            # Pad overlap image to obtain rotated bounding boxes even in cases when
            # overlap reaches images border.
            pad = int(overlap.shape[0] / 4)
            overlap_pad = np.pad(overlap, [[pad, pad], [pad, pad]])
            overlap_pad = (overlap_pad * 255).astype("uint8")

            # Get contour of overlap
            cnt, _ = cv2.findContours(
                overlap_pad, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE,
            )

            # There are nearly always multiple contours, some existing of only a few
            # points which we don't want to include in gradient blending. Hence, we
            # filter actual contours rather than line-like artefacts, check function
            # for criteria.
            actual_cnts = [np.squeeze(c) for c in cnt if is_valid_contour(c)]

            # In case of multiple valid contours, create gradient for each and sum
            if len(actual_cnts) > 1:
                all_grads = []
                all_grads_rev = []

                for c in actual_cnts:
                    all_contours.append(c)
                    bbox = cv2.minAreaRect(c)

                    # Get the gradient and its reverse
                    if is_horizontal:
                        grad, grad_rev = get_gradients(
                            bbox=bbox,
                            q1_mask=q1_mask,
                            overlap=overlap,
                            direction="horizontal",
                            pad=pad,
                        )

                    elif is_vertical:
                        grad, grad_rev = get_gradients(
                            bbox=bbox,
                            q1_mask=q1_mask,
                            overlap=overlap,
                            direction="vertical",
                            pad=pad,
                        )

                    all_grads.append(grad)
                    all_grads_rev.append(grad_rev)

                all_grad = np.sum(all_grads, axis=0)
                all_grad_rev = np.sum(all_grads_rev, axis=0)

                # Save the gradients
                gradients[(q1_name, q2_name)] = all_grad
                gradients[(q2_name, q1_name)] = all_grad_rev

            # In case of only 1 valid contour
            elif len(actual_cnts) == 1:
                c = np.squeeze(actual_cnts)
                all_contours.append(c)
                bbox = cv2.minAreaRect(c)

                # Get the gradient and its reverse
                if is_horizontal:
                    all_grad, all_grad_rev = get_gradients(
                        bbox=bbox,
                        q1_mask=q1_mask,
                        overlap=overlap,
                        direction="horizontal",
                        pad=pad,
                    )

                elif is_vertical:
                    all_grad, all_grad_rev = get_gradients(
                        bbox=bbox,
                        q1_mask=q1_mask,
                        overlap=overlap,
                        direction="vertical",
                        pad=pad,
                    )

                # Save the gradients
                gradients[(q1_name, q2_name)] = all_grad
                gradients[(q2_name, q1_name)] = all_grad_rev

            # Rare case when there is 1 contour but this contour is not valid and
            # basically an artefact. In this case we treat this as nonoverlap.
            else:
                is_overlap_list[-1] = False

    # Sum all non overlapping parts
    all_nonoverlap = np.sum(list(nonoverlap.values()), axis=0).astype("uint8")

    # Sum all overlapping parts relative to their gradient
    if True in is_overlap_list:
        grad_fragments = [images[str(j[0])] for j in gradients.keys()]
        all_overlap = np.sum(
            [
                (g[:, :, np.newaxis] * gq).astype("uint8")
                for g, gq in zip(gradients.values(), grad_fragments)
            ],
            axis=0,
        )
    else:
        all_overlap = np.zeros_like(all_nonoverlap)

    # Combine both parts and get copy for postprocessing
    final_image = all_nonoverlap + all_overlap
    final_image_edit = copy.deepcopy(final_image)

    # Check if there was any overlap between fragments
    if True in is_overlap_list:

        # Indices for getting patches
        p1, p2 = int(np.floor(patch_size_mean / 2)), int(np.ceil(patch_size_mean / 2))

        # Loop over all contours
        for cnt in all_contours:

            # Loop over points in each contour. Since we loop over a 3x3 grid for each
            # point we can skip half of the points.
            for pt in cnt[::2]:

                xrange = np.arange(-1, 2)
                yrange = np.arange(-1, 2)
                grid = np.meshgrid(xrange, yrange)

                for xx, yy in zip(grid[0].ravel(), grid[1].ravel()):

                    # Get x, y value of pixel to analyze.
                    x = (
                        pt[1] - pad + xx
                        if pt[1] - pad + xx < final_image.shape[0]
                        else final_image.shape[0] - 1
                    )
                    y = (
                        pt[0] - pad + yy
                        if pt[0] - pad + yy < final_image.shape[1]
                        else final_image.shape[1] - 1
                    )

                    # Get lower and upper bounds for the patch. Avoid potential negative
                    # indexing or indexing out of bounds.
                    x_lb = x - p1 if x - p1 > 0 else 0
                    x_ub = (
                        x + p2
                        if x + p2 < final_image.shape[0]
                        else final_image.shape[0] - 1
                    )
                    y_lb = y - p1 if y - p1 > 0 else 0
                    y_ub = (
                        y + p2
                        if y + p2 < final_image.shape[1]
                        else final_image.shape[1] - 1
                    )

                    # Extract patch, compute median pixel value and insert in image
                    patch = final_image[x_lb:x_ub, y_lb:y_ub, :]
                    patch = patch.reshape(
                        int(patch.shape[0] * patch.shape[1]), patch.shape[2]
                    )
                    fill_val = np.median(patch, axis=0)
                    final_image_edit[x, y, :] = fill_val

    # Clipping may be necessary for areas where more than 2 fragments overlap
    final_image_edit = np.clip(final_image_edit, 0, 255).astype("uint8")

    # Set 0 values to nan for plotting in blend summary.
    if len(list(gradients.values())) > 0:
        final_grad = list(gradients.values())[0]
        final_grad[final_grad == 0] = np.nan
        valid_cnt = True
    else:
        final_grad = np.full(final_image_edit.shape, np.nan)
        valid_cnt = False

    overlapping_fragments = overlapping_fragments[0]

    return final_image_edit, final_grad, overlapping_fragments, valid_cnt
