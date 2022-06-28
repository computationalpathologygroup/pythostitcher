import copy
import numpy as np
import cv2


def fuse_images(images):
    """
    Custom function to merge overlapping quadrants into a visually appealing combined
    image using alpha blending. This is accomplished by the following steps:
    1. Compute areas of overlap between different quadrants
    2. Compute bounding box around the overlap
    3. Compute alpha gradient field over this bounding box
    4. Mask the gradient field with the area of overlap
    5. Apply the masked gradient field to original image intensity
    6. Sum all resulting (non)overlapping images to create the final image

    Inputs
        - Final colour image of all quadrants

    Output
        - Merged output image
    """

    im_a, im_b, im_c, im_d = images

    # Simple threshold to get tissue masks
    mask_a = (np.mean(im_a, axis=-1) > 0) * 1
    mask_b = (np.mean(im_b, axis=-1) > 0) * 1
    mask_c = (np.mean(im_c, axis=-1) > 0) * 1
    mask_d = (np.mean(im_d, axis=-1) > 0) * 1

    # Get plausible overlapping quadrants
    quadrant_names = ["A", "B", "C", "D"]
    combinations = ["AB", "AC", "BD", "CD"]

    # Create some lists for iterating
    image_list = [im_a, im_b, im_c, im_d]
    mask_list = [mask_a, mask_b, mask_c, mask_d]
    total_mask = mask_a + mask_b + mask_c + mask_d
    all_contours = []

    # Create some dicts
    images = dict()
    masks = dict()
    gradients = dict()
    overlaps = dict()
    nonoverlap = dict()

    for name, im, mask in zip(quadrant_names, image_list, mask_list):
        images[name] = im
        masks[name] = mask

    # Some postprocessing values
    patch_size_mean = 15

    for combi in combinations:
        # Get names
        q1_name = combi[0]
        q2_name = combi[1]

        # Get quadrants
        q1_image = images[q1_name]
        q2_image = images[q2_name]
        q1_mask = masks[q1_name]
        q2_mask = masks[q2_name]

        # Compute non overlapping part of quadrants
        only_q1 = q1_image * (total_mask == 1)[:, :, np.newaxis]
        only_q2 = q2_image * (total_mask == 1)[:, :, np.newaxis]
        nonoverlap[q1_name] = only_q1
        nonoverlap[q2_name] = only_q2

        # Compute overlapping part of quadrants
        overlap = ((q1_mask + q2_mask) == 2) * 1
        overlaps[combi] = overlap

        # Compute bbox around overlap
        cnt, _ = cv2.findContours(
            (overlap * 255).astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        # Check if overlap exists. If so, only keep the largest area of overlap
        is_overlap = bool(cnt)
        if is_overlap:
            cnt = np.squeeze(max(cnt, key=cv2.contourArea))

        # Can only compute bounding box around contour when the contour is longer than
        # one single 2D point
        if len(cnt) > 2:
            all_contours.append(cnt)
            bbox = cv2.minAreaRect(cnt)

            # Extract bbox params
            bbox_center = bbox[0]
            angle = bbox[2]

            # OpenCV defines angles in the cv2.minAreaRect function between [0, 90] but
            # instead of a rotation of 0-90 degrees we can also rescale it to [0, 45] and
            # swap width/height.
            if angle > 45:
                angle = 90 - angle

            # Prepopulate gradient field
            gradient_2d = np.zeros_like(q1_mask).astype("float")

            # Check if overlap is between horizontally or vertically aligned quadrants
            is_horizontal = (sorted([q1_name, q2_name]) == ["A", "B"]) or (
                (sorted([q1_name, q2_name]) == ["C", "D"])
            )
            if is_horizontal:

                # See comment in line 92. With angles closer to 90 than to 0 we swap
                # width/height.
                if bbox[2] < 45:
                    width = int(bbox[1][0])
                    height = int(bbox[1][1])
                else:
                    width = int(bbox[1][1])
                    height = int(bbox[1][0])

                # Get slicing locations
                xmin = int(bbox_center[0] - 0.5 * width)
                xmax = xmin + width
                ymin = int(bbox_center[1] - 0.5 * height)
                ymax = ymin + height

                # Create 2d gradient
                gradient_1d = np.linspace(1, 0, width)
                gradient_2d_fill = np.tile(gradient_1d, (height, 1))

                # Rotate the gradient along its primary axis
                gradient_2d[ymin:ymax, xmin:xmax] = gradient_2d_fill
                rot_mat = cv2.getRotationMatrix2D(
                    center=bbox_center, angle=-angle, scale=1
                )
                gradient_2d = cv2.warpAffine(
                    gradient_2d, rot_mat, dsize=gradient_2d.shape[::-1]
                )
                gradients[combi] = gradient_2d * overlap

                # Compute the reverse gradient for scaling the other quadrant
                gradient_2d = np.zeros_like(q1_mask).astype("float")
                gradient_2d[ymin:ymax, xmin:xmax] = np.fliplr(gradient_2d_fill)
                rot_mat = cv2.getRotationMatrix2D(
                    center=bbox_center, angle=-angle, scale=1
                )
                gradient_2d_warp = cv2.warpAffine(
                    gradient_2d, rot_mat, dsize=gradient_2d.shape[::-1]
                )
                gradients[combi[::-1]] = gradient_2d_warp * overlap

            # Check if overlap is between horizontally or vertically aligned quadrants
            is_vertical = (sorted([q1_name, q2_name]) == ["A", "C"]) or (
                (sorted([q1_name, q2_name]) == ["B", "D"])
            )
            if is_vertical:

                # See comment in line 92. With angles closer to 90 than to 0 we swap
                # width/height.
                if bbox[2] < 45:
                    width = int(bbox[1][0])
                    height = int(bbox[1][1])
                else:
                    width = int(bbox[1][1])
                    height = int(bbox[1][0])

                # Get slicing locations
                xmin = int(bbox_center[0] - 0.5 * width)
                xmax = xmin + width
                ymin = int(bbox_center[1] - 0.5 * height)
                ymax = ymin + height

                # Create 2d gradient
                gradient_1d = np.linspace(1, 0, height)
                gradient_2d_fill = np.tile(gradient_1d, (width, 1))
                gradient_2d_fill = np.transpose(gradient_2d_fill)

                # Rotate the gradient along its primary axis
                gradient_2d[ymin:ymax, xmin:xmax] = gradient_2d_fill
                rot_mat = cv2.getRotationMatrix2D(
                    center=bbox_center, angle=-angle, scale=1
                )
                gradient_2d = cv2.warpAffine(
                    gradient_2d, rot_mat, dsize=gradient_2d.shape[::-1]
                )
                gradients[combi] = gradient_2d * overlap

                # Compute the reverse gradient for scaling the other quadrant
                gradient_2d = np.zeros_like(q1_mask).astype("float")
                gradient_2d[ymin:ymax, xmin:xmax] = np.flipud(gradient_2d_fill)
                rot_mat = cv2.getRotationMatrix2D(
                    center=bbox_center, angle=-angle, scale=1
                )
                gradient_2d_warp = cv2.warpAffine(
                    gradient_2d, rot_mat, dsize=gradient_2d.shape[::-1]
                )
                gradients[combi[::-1]] = gradient_2d_warp * overlap

    # Sum all non overlapping parts
    all_nonoverlap = np.sum(list(nonoverlap.values()), axis=0).astype("uint8")

    # Sum all overlapping parts relative to their gradient
    if is_overlap:
        gradient_quadrants = [images[str(j[0])] for j in gradients.keys()]
        all_overlap = np.sum(
            [
                (g[:, :, np.newaxis] * gq).astype("uint8")
                for g, gq in zip(gradients.values(), gradient_quadrants)
            ],
            axis=0,
        )
    else:
        all_overlap = np.zeros_like(all_nonoverlap)

    # Combine both parts
    final_image = all_nonoverlap + all_overlap

    ## Postprocess the final image to reduce stitching artefacts.
    final_image_edit = copy.deepcopy(final_image)

    # Loop over all contours
    if is_overlap:

        # Indices for getting patches
        p1, p2 = int(np.floor(patch_size_mean / 2)), int(np.ceil(patch_size_mean / 2))

        for cnt in all_contours:

            # Loop over all points in each contour
            for pt in cnt:
                # Replace each pixel value with the average value in a NxN neighbourhood
                # to reduce stitching artefacts
                x, y = pt[1], pt[0]
                patch = final_image[x - p1 : x + p2, y - p1 : y + p2, :]
                fill_val = np.mean(np.mean(patch, axis=0), axis=0)
                final_image_edit[x, y, :] = fill_val

    # Clipping may be necessary for areas where more than 2 quadrants overlap
    final_image_edit = np.clip(final_image_edit, 0, 255).astype("uint8")

    return final_image_edit