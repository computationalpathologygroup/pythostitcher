import pathlib
import matplotlib.pyplot as plt
import logging

from .jigsawnet_utils import *


def jigsawnet_scoring(parameters):
    """
    Main function for single testing mode.
    """

    parameters["log"].log(parameters["my_level"], f" - scoring pairs with JigsawNet")

    # Temporarily set logging to error to prevent annoying TF 1/2 warnings
    parameters["log"].setLevel(logging.ERROR)

    # Required for JigsawNet
    tf.compat.v1.disable_v2_behavior()

    # JigsawNet parameters
    K = parameters["JSN_Hyperparameters"]["learner_num"]  # the number of learner for boost training
    checkpoint_root = pathlib.Path(parameters["weights_jigsawnet"])
    max_expand_threshold = parameters["max_expand_threshold"]

    # Load the model
    with open(checkpoint_root.joinpath("alpha.txt")) as f:
        for line in f:
            line = line.rstrip()
            if line[0] != "#":
                line = line.split()
                Alpha = [float(x) for x in line]

    # Load model and evaluator
    net = JigsawNetWithROI(params=parameters["JSN_Hyperparameters"])
    evaluator = SingleTest(checkpoint_root=checkpoint_root, K=5, net=net, is_training=False)

    # Process alignments
    alignment_file = parameters["save_dir"].joinpath("configuration_detection", "alignments.txt")
    alignments = Alignment2d(alignment_file)

    # Process background color file
    bg_color_file = parameters["save_dir"].joinpath("configuration_detection", "bg_color.txt")
    with open(bg_color_file) as f:
        for line in f:
            line = line.split()
            if line:
                bg_color = [int(i) for i in line]
                bg_color = bg_color[::-1]

    # Create lists to save results
    all_inference_images = []
    all_inference_prob = []
    all_inference_bbox = []
    all_inference_v1 = []
    all_inference_v2 = []
    all_inference_trans = []

    print("Performing inference with JigsawNet")
    # Inference loop
    for count, alignment in enumerate(alignments.data):
        print(f" - scoring image pair {count+1}/{len(alignments.data)}")

        # Get data from pairwise alignment
        v1 = alignment.frame1
        v2 = alignment.frame2
        trans = alignment.transform
        raw_stitch_line = alignment.stitchLine

        # Load images to be stitched
        image1_path = parameters["save_dir"].joinpath(
            "configuration_detection", f"fragment{str(v1+1)}.png"
        )
        image1 = cv2.imread(str(image1_path))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        image2_path = parameters["save_dir"].joinpath(
            "configuration_detection", f"fragment{str(v2+1)}.png"
        )
        image2 = cv2.imread(str(image2_path))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        # Get the fused image with the pairwise alignment
        fusion_output = FusionImage(image1, image2, trans, bg_color)
        path_img, point_tform, overlap_ratio, transform_offset = fusion_output

        # Resize image to fit network [160, 160]
        resized_path_img = cv2.resize(
            path_img,
            (
                parameters["JSN_Hyperparameters"]["height"],
                parameters["JSN_Hyperparameters"]["width"],
            ),
        )
        net.evaluate_image = resized_path_img
        all_inference_images.append(resized_path_img)

        # Compute the ratio of the image where the attention should focus
        [
            new_min_row_ratio,
            new_min_col_ratio,
            new_max_row_ratio,
            new_max_col_ratio,
        ] = ConvertRawStitchLine2BBoxRatio(
            raw_stitch_line,
            path_img,
            trans,
            transform_offset,
            max_expand_threshold=max_expand_threshold,
        )
        net.roi_box = [new_min_row_ratio, new_min_col_ratio, new_max_row_ratio, new_max_col_ratio]
        all_inference_bbox.append(net.roi_box)

        # Make prediction
        preds, probs = next(evaluator)

        # Loop over all models in ensemble
        for i in range(K):
            if preds[i] == 1:
                preds[i] = 1
            else:
                preds[i] = -1

        # Get the final prediction based on the ensemble.
        correct_probs = []
        for i in range(len(probs)):
            correct_probs.append(probs[i][1])

        # Compute the correct probability based on a weighting per model
        correct_probability = np.sum(np.multiply(correct_probs, Alpha)) / np.sum(Alpha)

        # Save results for later writing to txt file
        all_inference_v1.append(v1)
        all_inference_v2.append(v2)
        all_inference_trans.append(trans)
        all_inference_prob.append(correct_probability)

    # The previous method only writes down the JigsawNet result if the final
    # class equals 1, which is computed by taking a weighted average of the
    # different models of the ensemble (NOTE: not equal to avg_prob>0.5!).
    # By lowering the threshold we introduce more false positives which may
    # be required in our case as stitches will never be near perfect.
    correct_threshold = 1e-20

    # Create filtered alignments file to write results
    with open(
        parameters["save_dir"].joinpath("configuration_detection", "filtered_alignments.txt"), "w"
    ) as f1:
        for v1, v2, prob, trans in zip(
            all_inference_v1, all_inference_v2, all_inference_prob, all_inference_trans
        ):
            if correct_probability > correct_threshold:
                f1.write("%d\t%d\t%f\t0\n" % (v1, v2, prob))
                f1.write(
                    "%f %f %f\n%f %f %f\n0 0 1\n"
                    % (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2])
                )
    f1.close()

    # Save figure with all probabilites
    plt.figure(figsize=(8, 14))
    plt.suptitle("Image pairs with JigsawNet score and attention box\n", fontsize=20)
    for c, (im, pred, bbox) in enumerate(
        zip(all_inference_images, all_inference_prob, all_inference_bbox), 1
    ):

        # Get bbox with attention
        im_size = resized_path_img.shape[0]
        [new_min_row_ratio, new_min_col_ratio, new_max_row_ratio, new_max_col_ratio] = bbox
        bbox_min_col = im_size * new_min_col_ratio
        bbox_max_col = im_size * new_max_col_ratio
        bbox_min_row = im_size * new_min_row_ratio
        bbox_max_row = im_size * new_max_row_ratio
        bbox_coords = np.array(
            [
                [bbox_min_col, bbox_min_row],
                [bbox_min_col, bbox_max_row],
                [bbox_max_col, bbox_max_row],
                [bbox_max_col, bbox_min_row],
            ]
        )
        test = np.vstack([bbox_coords, bbox_coords[0]])

        plt.subplot(6, 4, c)
        plt.title(f"pred: {pred:.4f}")
        plt.imshow(im)
        plt.plot(test[:, 0], test[:, 1], c="g", linewidth=3)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        parameters["save_dir"].joinpath(
            "configuration_detection", "checks", f"jigsawnet_pred_expand{max_expand_threshold}.png"
        )
    )
    plt.close()

    # Reset graphs required when performing reassembly for multiple cases
    tf.compat.v1.reset_default_graph()
    parameters["log"].setLevel(logging.WARNING)

    return


def SingleTest(checkpoint_root, K, net, is_training=False):
    """
    Main function for evaluating own image data.

    Inputs:
        - checkpoint_root: path to weights
        - K: number of models in ensemble
        - net: the model to use (jigsawnet)
        - is_training: whether the model is in training mode

    Output:
        - Jigsawnet prediction
    """

    input = tf.keras.Input(
        shape=[net.params["height"], net.params["width"], net.params["depth"]], dtype=tf.float32
    )
    roi_box = tf.keras.Input(shape=[4], dtype=tf.float32)

    logits = net._inference(input, roi_box, is_training)
    probability = tf.nn.softmax(logits)

    # Get all models and restore the weights
    sessions = []
    saver = tf.compat.v1.train.Saver(max_to_keep=10)

    for i in range(K):
        check_point = os.path.join(checkpoint_root, "g%d" % i)
        sess = tf.compat.v1.Session()
        sess_init_op = tf.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(sess_init_op)
        saver.restore(sess, tf.train.latest_checkpoint(check_point))
        sessions.append(sess)

    # Inference on JigsawNet. Note how this is only performed with batch_size=1. Perhaps
    # quite some potential to speed this up.
    while not net.close:
        if len(np.shape(net.evaluate_image)) < 4:
            net.evaluate_image = np.reshape(
                net.evaluate_image,
                [1, net.params["height"], net.params["width"], net.params["depth"]],
            )
        if len(np.shape(net.roi_box)) < 2:
            net.roi_box = np.reshape(net.roi_box, [1, 4])

        # Save predictions and probabilities
        preds = []
        probs = []  # correct and incorrect probability
        for i in range(K):
            pred, prob = sessions[i].run(
                [net.pred, probability], feed_dict={input: net.evaluate_image, roi_box: net.roi_box}
            )
            pred = pred[0]
            prob = prob[0]
            preds.append(pred)
            probs.append(prob)
        yield preds, probs

    # Close sessions after inference
    for sess in sessions:
        sess.close()

    return
