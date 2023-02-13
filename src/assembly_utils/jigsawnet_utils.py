import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.training import moving_averages
import operator
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


class JigsawNetWithROI:
    def __init__(self, params):
        # hyperparameters
        self.params = params
        self.evaluate_image = None
        self.roi_box = None
        self.close = False


    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'session'):
            self.session.close()

    def _variable_summaries(self, v, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.compat.v1.name_scope(name):
            tf.compat.v1.summary.histogram('histogram', v)

    def _pooling_layer(self, input, name_scope):
        with tf.compat.v1.variable_scope(name_scope):
            x = tf.nn.max_pool2d(input=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name="Pooling_layer")
        return x

    def _roi_pooling_layer(self, input, name_scope, box):
        """output 4x4"""
        with tf.compat.v1.variable_scope(name_scope):
            roi_out_size = tf.constant([4, 4])
            box_indices = tf.range(start=0, limit=tf.shape(input=input)[0], dtype=tf.int32)
            self.box_indices = box_indices
            align_roi = tf.image.crop_and_resize(image=input, boxes=box, box_indices=box_indices, crop_size=roi_out_size, method='bilinear', name='align_roi')
        return align_roi


    def _BN(self, input, filter_num, is_training, decay=0.99, beta_name="BN_beta", gamma_name="BN_gamma",
             mov_avg_name="BN_moving_avg", mov_var_name="BN_moving_var"):
        """
        Note: batch normalization may have negative effect on performance if the mini-batch size is small.
        see https://arxiv.org/pdf/1702.03275.pdf
        Also, batch normalization has different behavior between training and testing
        """
        beta = tf.compat.v1.get_variable(name=beta_name, shape=filter_num, dtype=tf.float32,
                               initializer=tf.compat.v1.constant_initializer(0.0))
        gamma = tf.compat.v1.get_variable(name=gamma_name, shape=filter_num, dtype=tf.float32,
                                initializer=tf.compat.v1.constant_initializer(1.0))
        moving_avg = tf.compat.v1.get_variable(name=mov_avg_name, shape=filter_num, dtype=tf.float32,
                                     initializer=tf.compat.v1.constant_initializer(0.0), trainable=False)
        moving_var = tf.compat.v1.get_variable(name=mov_var_name, shape=filter_num, dtype=tf.float32,
                                     initializer=tf.compat.v1.constant_initializer(1.0), trainable=False)

        control_inputs = []
        if is_training:
            mean, var = tf.nn.moments(x=input, axes=[0, 1, 2])
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, mean, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]

        else:
            '''during testing, should use moving avg and var'''
            mean = moving_avg
            var = moving_var
        with tf.control_dependencies(control_inputs):
            x = tf.nn.batch_normalization(input, mean, var, offset=beta, scale=gamma, variance_epsilon=1e-3)
        return x

    def _conv_layer(self, input, is_training):
        # initalize some parameters
        stride = 1
        filter_shape = [3, 3, self.params['depth'], 8]

        '''
            operation
        '''
        with tf.compat.v1.variable_scope("init_conv_layer"):
            # 1. convolution
            filter_weights = tf.compat.v1.get_variable(name='ConvLayer_filter', shape=filter_shape,
                                             initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                             regularizer=tf.keras.regularizers.l2(
                                                 l=0.5 * (self.params['weight_decay'])))
            bias = tf.compat.v1.get_variable(name="ConvLayer_biases", initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                   shape=[filter_shape[3]])
            x = tf.nn.conv2d(input=input, filters=filter_weights, strides=[1, stride, stride, 1], padding="SAME")
            x = tf.nn.bias_add(x, bias)

            # 2. BN
            x = self._BN(input=x, filter_num=filter_shape[3], is_training=is_training)

            self.first_feature_map = x

            # 3. relu
            x = tf.nn.relu(x)

        return x

    def _residual_layer(self, input, filter_in, filter_out, is_training, name_scope):
        # initalize some parameters
        stride = 1
        filter_shape1 = [3, 3, filter_in, filter_out]
        filter_shape2 = [3, 3, filter_out, filter_out]
        '''
            operation
        '''
        with tf.compat.v1.variable_scope(name_scope):
            # 1. convolution
            filter_weights1 = tf.compat.v1.get_variable(name='ResLayer_filter1', shape=filter_shape1,
                                              initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                              regularizer=tf.keras.regularizers.l2(
                                                  l=0.5 * (self.params['weight_decay'])))
            bias1 = tf.compat.v1.get_variable(name="ResLayer_filter1_biases", initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    shape=[filter_shape1[3]])
            x = tf.nn.conv2d(input=input, filters=filter_weights1, strides=[1, stride, stride, 1], padding="SAME")
            x = tf.nn.bias_add(x, bias1)
            self._variable_summaries(filter_weights1, 'ResLayer_filter1')
            # 2. BN
            x = self._BN(input=x, filter_num=filter_shape1[3], is_training=is_training, beta_name="ResLayer_BN_beta1",
                          gamma_name="ResLayer_BN_gamma1", mov_avg_name="ResLayer_BN_mov_avg1",
                          mov_var_name="ResLayer_BN_mov_var1")
            ResLayer_BN_beta1 = [v for v in tf.compat.v1.global_variables() if v.name == name_scope + "/ResLayer_BN_beta1:0"][0]
            ResLayer_BN_gamma1 = [v for v in tf.compat.v1.global_variables() if v.name == name_scope + "/ResLayer_BN_gamma1:0"][0]
            self._variable_summaries(ResLayer_BN_beta1, 'ResLayer_BN_beta1')
            self._variable_summaries(ResLayer_BN_gamma1, 'ResLayer_BN_gamma1')
            # 3. relu
            x = tf.nn.relu(x)
            # 4. convolution
            filter_weights2 = tf.compat.v1.get_variable(name='ResLayer_filter2', shape=filter_shape2,
                                              initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                              regularizer=tf.keras.regularizers.l2(
                                                  l=0.5 * (self.params['weight_decay'])))
            bias2 = tf.compat.v1.get_variable(name="ResLayer_filter2_biases", initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    shape=[filter_shape2[3]])
            x = tf.nn.conv2d(input=x, filters=filter_weights2, strides=[1, stride, stride, 1], padding="SAME")
            x = tf.nn.bias_add(x, bias2)
            # 5. BN
            x = self._BN(input=x, filter_num=filter_shape1[3], is_training=is_training, beta_name="ResLayer_BN_beta2",
                          gamma_name="ResLayer_BN_gamma2", mov_avg_name="ResLayer_BN_mov_avg2",
                          mov_var_name="ResLayer_BN_mov_var2")

            # 6. skip connection
            if filter_in != filter_out:
                # skip conv
                filter_weights1 = tf.compat.v1.get_variable(name='ResLayer_skip_filter', shape=filter_shape1,
                                                  initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                                  regularizer=tf.keras.regularizers.l2(
                                                      l=0.5 * (self.params['weight_decay'])))
                bias1 = tf.compat.v1.get_variable(name="ResLayer_skip_filter_biases",
                                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), shape=[filter_shape1[3]])
                skip_out = tf.nn.conv2d(input=input, filters=filter_weights1, strides=[1, stride, stride, 1], padding="SAME")
                skip_out = tf.nn.bias_add(skip_out, bias1)
                # skip BN
                skip_out = self._BN(input=skip_out, filter_num=filter_shape1[3], is_training=is_training,
                                     beta_name="ResLayer_skip_BN_beta1", gamma_name="ResLayer_skip_BN_gamma1",
                                     mov_avg_name="ResLayer_skip_BN_mov_avg1", mov_var_name="ResLayer_skip_BN_mov_var1")

                x = x + skip_out
            else:
                x = x + input
            # 7. relu
            x = tf.nn.relu(x)

        return x

    def _classify(self, geometric_feature, roi_feature):
        input_roi_w = 4
        input_roi_h = 4
        input_geo_w = 10
        input_geo_h = 10
        input_d = 128
        fc1_geo_dim = 32
        fc1_roi_dim = 32
        fc2_dim = 2
        '''operation'''
        with tf.compat.v1.variable_scope("value_head"):
            # 4. fc1 for geometric feature map
            flat_size = input_d * input_geo_h * input_geo_w
            geometric_feature_flat = tf.reshape(geometric_feature, [-1, flat_size])
            fc_geo_w = tf.compat.v1.get_variable(name='ValueLayer_fc1_geo_w', shape=[flat_size, fc1_geo_dim],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    regularizer=tf.keras.regularizers.l2(l=0.5 * (self.params['weight_decay'])))
            fc_geo_b = tf.compat.v1.get_variable(name='ValueLayer_fc1_geo_bias', shape=[fc1_geo_dim], initializer=tf.compat.v1.zeros_initializer())
            x_geo = tf.matmul(geometric_feature_flat, fc_geo_w) + fc_geo_b

            # 5. fc1 for roi feature map
            flat_size = input_d * input_roi_h * input_roi_w
            roi_feature_flat = tf.reshape(roi_feature, [-1, flat_size])
            fc_roi_w = tf.compat.v1.get_variable(name='ValueLayer_fc1_roi_w', shape=[flat_size, fc1_roi_dim],
                                       initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                       regularizer=tf.keras.regularizers.l2(l=0.5 * (self.params['weight_decay'])))
            fc_roi_b = tf.compat.v1.get_variable(name='ValueLayer_fc1_roi_bias', shape=[fc1_roi_dim],
                                       initializer=tf.compat.v1.zeros_initializer())
            x_roi = tf.matmul(roi_feature_flat, fc_roi_w) + fc_roi_b

            # concatenate
            x = tf.concat([x_geo, x_roi], axis=1)

            # 5. fully connection 2
            fc_w2 = tf.compat.v1.get_variable(name='ValueLayer_fc_w2', shape=[fc1_geo_dim+fc1_roi_dim, fc2_dim],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    regularizer=tf.keras.regularizers.l2(l=0.5 * (self.params['weight_decay'])))
            fc_b2 = tf.compat.v1.get_variable(name='ValueLayer_fc_bias2', shape=[fc2_dim], initializer=tf.compat.v1.zeros_initializer())
            x = tf.matmul(x, fc_w2) + fc_b2
            self.fc2_shape = tf.shape(input=x)

        return x

    def _classifyOnlyROIFeature(self, roi_feature):
        input_roi_w = 4
        input_roi_h = 4
        input_d = 128
        fc1_roi_dim = 64
        fc2_dim = 2
        '''operation'''
        with tf.compat.v1.variable_scope("value_head"):
            # 5. fc1 for roi feature map
            flat_size = input_d * input_roi_h * input_roi_w
            roi_feature_flat = tf.reshape(roi_feature, [-1, flat_size])
            fc_roi_w = tf.compat.v1.get_variable(name='ValueLayer_fc1_roi_w', shape=[flat_size, fc1_roi_dim],
                                       initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                       regularizer=tf.keras.regularizers.l2(l=0.5 * (self.params['weight_decay'])))
            fc_roi_b = tf.compat.v1.get_variable(name='ValueLayer_fc1_roi_bias', shape=[fc1_roi_dim],
                                       initializer=tf.compat.v1.zeros_initializer())
            x_roi = tf.matmul(roi_feature_flat, fc_roi_w) + fc_roi_b

            # concatenate
            x = x_roi

            # 5. fully connection 2
            fc_w2 = tf.compat.v1.get_variable(name='ValueLayer_fc_w2', shape=[fc1_roi_dim, fc2_dim],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    regularizer=tf.keras.regularizers.l2(l=0.5 * (self.params['weight_decay'])))
            fc_b2 = tf.compat.v1.get_variable(name='ValueLayer_fc_bias2', shape=[fc2_dim], initializer=tf.compat.v1.zeros_initializer())
            x = tf.matmul(x, fc_w2) + fc_b2
            self.fc2_shape = tf.shape(input=x)
        return x

    def _inference(self, input, roi_box, is_training):
        '''

        :param input: input image
        :param roi: [start row ratio, start col ratio, end row ratio, end col ratio]. e.g [0, 0, 0.5, 0.5]
        :return:
        '''

        x = self._conv_layer(input, is_training=is_training)
        self.block1_shape = tf.shape(input=x)
        x = self._pooling_layer(x, name_scope="init_conv_layer")

        name_scope = "residual_layer_0"
        x = self._residual_layer(x, filter_in=8, filter_out=8, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_1"
        x = self._residual_layer(x, filter_in=8, filter_out=8, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_2"
        x = self._residual_layer(x, filter_in=8, filter_out=16, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_3"
        x = self._residual_layer(x, filter_in=16, filter_out=16, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_4"
        x = self._residual_layer(x, filter_in=16, filter_out=16, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_5"
        x = self._residual_layer(x, filter_in=16, filter_out=32, name_scope=name_scope, is_training=is_training)
        x = self._pooling_layer(x, name_scope="first_6_residual")

        name_scope = "residual_layer_6"
        x = self._residual_layer(x, filter_in=32, filter_out=32, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_7"
        x = self._residual_layer(x, filter_in=32, filter_out=32, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_8"
        x = self._residual_layer(x, filter_in=32, filter_out=64, name_scope=name_scope, is_training=is_training)
        x = self._pooling_layer(x, name_scope="second_3_residual")

        name_scope = "residual_layer_9"
        x = self._residual_layer(x, filter_in=64, filter_out=64, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_10"
        x = self._residual_layer(x, filter_in=64, filter_out=64, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_11"
        x = self._residual_layer(x, filter_in=64, filter_out=128, name_scope=name_scope, is_training=is_training)
        # geometric_feature = self._pooling_layer(x, name_scope="third_3_residual")
        roi_feature = self._roi_pooling_layer(x, name_scope="roi_pooling", box=roi_box)

        # x = self._classify(geometric_feature, roi_feature)
        x = self._classifyOnlyROIFeature(roi_feature)
        self.pred = tf.argmax(input=x, axis=1, name="prediction")
        return x

    def _loss(self, pred, target_value, weights=None, data_ids=None):
        if weights!=None:
            corresponding_weights = tf.gather(weights, data_ids)
            cross_e = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target_value), logits=pred, name='entropy')
            weighted_cross_e = tf.multiply(corresponding_weights, cross_e)
            entropy_loss = tf.reduce_mean(input_tensor=weighted_cross_e)
        else:
            entropy_loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target_value), logits=pred, name='entropy'))
        tf.compat.v1.summary.scalar('cross_entropy_loss', entropy_loss)
        reg_loss = tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
        tf.compat.v1.summary.scalar('reg_loss', reg_loss)

        losses = {}
        losses['value_loss'] = entropy_loss
        losses['reg_loss'] = reg_loss

        return losses

    def _optmization(self, losses, global_step):
        with tf.compat.v1.name_scope('training'):
            # learning_rate = tf.train.exponential_decay(self.params['learning_rate'], global_step, 15000, 0.1, staircase=True)
            learning_rate = self.params['learning_rate']
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = opt.compute_gradients(losses['value_loss'] + losses['reg_loss'])
            tf.compat.v1.summary.scalar('total_loss', losses['value_loss'] + losses['reg_loss'])

            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

        return opt_op

    ##########################################################################
    #   interface functions. Training and Testing
    ##########################################################################
    def train(self, input, roi_box, target, tensorboard_dir, checkpoint_dir, is_training):
        target_value = target
        gt_classification = tf.argmax(input=target, axis=1, name="gt_classification")
        logits = self._inference(input, roi_box, is_training)
        with tf.compat.v1.name_scope('accuracy'):
            correct_prediction = tf.equal(self.pred, gt_classification)
            accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
        tf.compat.v1.summary.scalar('accuracy', tf.reduce_mean(input_tensor=accuracy))

        losses = self._loss(logits, target_value)

        global_step = tf.Variable(0, trainable=False, name="global_step")
        opt_op = self._optmization(losses=losses, global_step=global_step)
        merged = tf.compat.v1.summary.merge_all()

        sess_init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        saver = tf.compat.v1.train.Saver(max_to_keep=2)

        with tf.compat.v1.Session() as sess:
            sess.run(sess_init_op)
            tensorboard_writer = tf.compat.v1.summary.FileWriter(tensorboard_dir, sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(coord=coord)

            while global_step.eval() < NNHyperparameters['total_training_step']:
                if global_step.eval() % 1000 == 0 and global_step.eval() != 0:
                    print("Check point...", end='')
                    saver.save(sess, checkpoint_dir, global_step=global_step)
                    print("Done")

                print("current step: %d" % tf.compat.v1.train.global_step(sess, global_step))
                if (global_step.eval() + 1) % 10 == 0:
                    visualization = False
                else:
                    visualization = False
                if global_step.eval() % 10 == 0:
                    tensorboard_record = True
                else:
                    tensorboard_record = False

                if not visualization:
                    _, la, value_loss, acc, summary = sess.run([opt_op, target, losses['value_loss'], accuracy, merged])
                    if tensorboard_record:
                        tensorboard_writer.add_summary(summary, global_step.eval())
                else:
                    _, im, feature_map1, la, value_loss, acc, summary = sess.run(
                        [opt_op, input, self.first_feature_map, target, losses['value_loss'], accuracy, merged])
                    if tensorboard_record:
                        tensorboard_writer.add_summary(summary, global_step.eval())
                    cv2.imshow("state", im[0].astype(np.uint8))
                    FirstFestureMap = feature_map1[0]
                    for i in range(8):
                        layer = FirstFestureMap[:, :, i]
                        layer = np.reshape(layer, (layer.shape[0], layer.shape[1], 1))
                        layer_channel_3 = np.concatenate((layer, layer), axis=2)
                        layer_channel_3 = np.concatenate((layer_channel_3, layer), axis=2)
                        layer_channel_3 = cv2.resize(layer_channel_3, (256, 256))
                        cv2.imshow("feature map %d" % i, layer_channel_3)
                    cv2.waitKey()

                print("value_loss: " + str(value_loss))
                print("accuracy: " + str(acc))
                print("---------------------------")
            tensorboard_writer.close()
            print("session graph has saved to " + tensorboard_dir)

            print("Save final results...", end='')
            saver.save(sess, checkpoint_dir, global_step=global_step)
            print("Done!")

            coord.request_stop()
            coord.join(threads)


    # this function is used for evaluating single image with session persistence
    def singleTest(self, checkpoint_dir, is_training):
        ### DEBUG REMOVE ###
        """
        input = tf.compat.v1.placeholder(tf.float32, [None, self.params['height'], self.params['width'], self.params['depth']])
        roi_box = tf.compat.v1.placeholder(tf.float32, [None, 4])
        """
        ### DEBUG REMOVE ###

        ### DEBUG ADD ###
        input = tf.keras.Input(shape=[self.params['height'], self.params['width'], self.params['depth']], dtype=tf.float32)
        roi_box = tf.keras.Input(shape=[4], dtype=tf.float32)
        ### DEBUG ADD ###

        logits = self._inference(input, roi_box, is_training)
        probability = tf.nn.softmax(logits)
        saver = tf.compat.v1.train.Saver(max_to_keep=2)
        with tf.compat.v1.Session() as sess:
            if checkpoint_dir!=None:
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir + '/'))
                print("model restored!")
            else:
                sess_init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
                sess.run(sess_init_op)
            while not self.close:
                if len(np.shape(self.evaluate_image)) <4:
                    self.evaluate_image = np.reshape(self.evaluate_image, [1, self.params['height'], self.params['width'], self.params['depth']])
                if len(np.shape(self.roi_box))<2:
                    self.roi_box = np.reshape(self.roi_box, [1, 4])
                prediction, prob = sess.run([self.pred, probability], feed_dict={input: self.evaluate_image, roi_box:self.roi_box})

                yield prediction, prob
        yield None


    def batchTest(self, input, roi_box, target, checkpoint_dir, is_training):
        gt_classification = tf.argmax(input=target, axis=1, name="gt_classification")
        logits = self._inference(input, roi_box, is_training)
        with tf.compat.v1.name_scope('accuracy'):
            correct_prediction = tf.equal(self.pred, gt_classification)
            accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
        saver = tf.compat.v1.train.Saver(max_to_keep=2)
        sess_init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

        with tf.compat.v1.Session() as sess:
            sess.run(sess_init_op)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir + '/'))
            print("model restored!")
            coord = tf.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

            try:
                tp=0
                tn=0
                fp=0
                fn=0
                while True:
                    acc, pred, gt, im, la = sess.run([accuracy, self.pred, gt_classification, input, target])
                    for i in range(len(pred)):
                        if pred[i] == gt[i] and pred[i]==0:
                            tn+=1
                        if pred[i] == gt[i] and pred[i]==1:
                            tp+=1
                        if pred[i] !=gt[i] and pred[i] ==0:
                            fp+=1
                        if pred[i] !=gt[i] and pred[i] ==1:
                            fn+=1
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)

        print("tp, tn, fp, fn: %d, %d, %d, %d" %(tp, tn, fp, fn))



'''
This file save some common data structures
'''

class GtPose:
    def __init__(self, gt_filename):
        self.data = []
        with open(gt_filename) as f:
            all_line = [line.rstrip() for line in f]
            for index, line in enumerate(all_line):
                if index%2!=0:
                    m1, m2, m3, m4, m5, m6, m7, m8, m9 = [t(s) for t, s in zip((float, float, float, float, float, float, float, float, float), line.split())]
                    pose = np.array([[m1, m2, m3], [m4, m5, m6], [m7, m8, m9]])
                    self.data.append(pose)

        # transform ground truth to identity
        baseline = np.linalg.inv(self.data[0])
        for i in range(len(self.data)):
            self.data[i] = np.matmul(baseline, self.data[i])

class GtPoseMatrix:
    def __init__(self, gt_filename):
        self.data=list()
        with open(gt_filename) as f:
            all_line = [line.rstrip() for line in f]
            for i in range(len(all_line)//4):
                matrix = all_line[i*4:(i+1)*4]
                line1 = matrix[0]
                line2 = matrix[1]
                line3 = matrix[2]
                line4 = matrix[3]
                fragmentId = int(line1)
                assert i==fragmentId
                m1, m2, m3 = [t(s) for t, s in zip((float, float, float), line2.split())]
                m4, m5, m6 = [t(s) for t, s in zip((float, float, float), line3.split())]
                m7, m8, m9 = [t(s) for t, s in zip((float, float, float), line4.split())]
                pose = np.array([[m1, m2, m3], [m4, m5, m6], [m7, m8, m9]])
                self.data.append(pose)


class PoseContainer:
    def __init__(self, pose_list):
        self.data = pose_list

    def SaveToFile(self, filename):
        with open(filename, 'w') as f:
            for fragmentId, pose in self.data:
                data_item = "%d\t%d\n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n"%(fragmentId, fragmentId, pose[0, 0], pose[0, 1], pose[0, 2], pose[1, 0], pose[1, 1], pose[1, 2], pose[2, 0], pose[2, 1], pose[2, 2])
                f.write(data_item)

    def CompareWithGT(self, gtpose, t_threshold, r_threshold):
        evaluation = dict()
        all_successful = True
        for fragmentId, pose in self.data:
            gt_pose = gtpose.data[fragmentId]
            err_mat = np.matmul(pose, np.linalg.inv(gt_pose))

            theta = np.arccos(err_mat[0,0])*180/3.1415926
            t = np.sqrt(err_mat[0,2]**2+err_mat[1,2]**2)
            if theta<r_threshold and t<t_threshold:
                evaluation[fragmentId] = True
            else:
                evaluation[fragmentId] = False
                all_successful = False
        return evaluation, all_successful

class Transform2d:
    def __init__(self, v1=-1, v2=-1, score=-1, transform=np.identity(3), stitchLine=None):
        self.frame1 = v1
        self.frame2 = v2
        self.score = score
        self.transform = transform
        self.stitchLine = stitchLine

        # rank between frame1 and frame2
        self.rank = -1

class Alignment2d:
    def __init__(self, relative_transform_filename):
        self.data = []
        # for example, {'0 1': [0,1,2]} means from 0--1 to find data[0,1,2]
        self.mapIdpair2Transform = {}
        # for example, {'0 1 1': 0} means from 0--1 and 1st to find data[0]
        self.mapIdpairRank2Transform = {}
        # for example, {0: [0,1,2,100]} means from 0 to find data[0,1,2,100] in which either 0-x or x-0
        self.id2Transform = {}

        with open(relative_transform_filename) as f:
            all_line = [line.rstrip() for line in f]
            node_num = 0
            for line in all_line:
                if line[0:4] == "Node":
                    node_num+=1
                else:
                    data_str_list = line.split()
                    v1,v2,score, m1,m2,m3,m4,m5,m6,m7,m8,m9 = [t(s) for t,s in zip((int,int, float, float,float,float,float,float,float,float,float,float), data_str_list[0:12])]
                    transform = np.array([[m1,m2,m3], [m4,m5,m6], [m7,m8,m9]])

                    stitchLine = []
                    stitch_line_c = data_str_list[13:]
                    for i in range(len(stitch_line_c)//2):
                        col = float(stitch_line_c[i*2])
                        row = float(stitch_line_c[i*2+1])
                        stitchLine.append([row, col])
                    self.data.append(Transform2d(v1, v2, score, transform, stitchLine))


        self.data = sorted(self.data, key=operator.attrgetter('score'), reverse=True)
        self.data = sorted(self.data, key=operator.attrgetter('frame2'))
        self.data = sorted(self.data, key=operator.attrgetter('frame1'))

        for i, item in enumerate(self.data):
            idpair = '%d %d'%(item.frame1, item.frame2)
            if idpair in self.mapIdpair2Transform:
                self.mapIdpair2Transform[idpair].append(i)
            else:
                self.mapIdpair2Transform[idpair] = [i]
            if item.frame1 in self.id2Transform:
                self.id2Transform[item.frame1].append(i)
            else:
                self.id2Transform[item.frame1] = [i]
            if item.frame2 in self.id2Transform:
                self.id2Transform[item.frame2].append(i)
            else:
                self.id2Transform[item.frame2] = [i]

        for key,value in self.mapIdpair2Transform.items():
            for rank, index in enumerate(value):
                new_key = "%s %d"%(key, rank+1)
                self.mapIdpairRank2Transform[new_key] = index
                self.data[index].rank = rank+1

def ExpandROI(aligned_img, bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col, max_expand_threshold):
    """
    Create the ROI of the stitching edge on which the model should pay extra attention.
    """

    # Due to a np.ceil operation in a previous function the max col/row of the bounding box
    # can exceed the dimension of the image. To account for this, we reduce the max col/row
    # in case this happens to prevent indexing error later on.
    if bbox_max_col > aligned_img.shape[1]:
        bbox_max_col = aligned_img.shape[1]
    if bbox_max_row > aligned_img.shape[0]:
        bbox_max_row = aligned_img.shape[0]

    # Set boundaries as starting point for the ROI determination
    min_row1, min_col1, max_row1, max_col1 = bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col
    min_row2, min_col2, max_row2, max_col2 = bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col
    bg_thres = 0.99
    original_area = (max_row1-min_row1) * (max_col1-min_col1)

    # Determine whether we have a vertical/horizontal stitch
    temp_mask = ~np.all(aligned_img==0, axis=2)*1
    r, c = np.nonzero(temp_mask)
    height = np.max(r) - np.min(r)
    width = np.max(c) - np.min(c)

    h_factor = 50 if width < height else 1
    v_factor = 50 if height < width else 1

    ''' 1. try to move upper and lower boundary first'''
    # upper boundary move up
    for i in range(1, max_expand_threshold * v_factor):
        row = bbox_min_row-i
        if row<0:
            break

        # Only expand ROI if it doesn't contain too many background pixels
        bg_pixel_count = aligned_img[row, bbox_min_col:bbox_max_col].tolist().count([0, 0, 0])
        bg_pixel_ratio = bg_pixel_count/(bbox_max_col-bbox_min_col)

        if bg_pixel_ratio > bg_thres:
            break
        else:
            min_row1 = row

    # lower boundary move down
    for i in range(1, max_expand_threshold * v_factor):
        row = bbox_max_row + i
        if row>=aligned_img.shape[0]:
            break

        # Only expand ROI if it doesn't contain too many background pixels
        bg_pixel_count = aligned_img[row, bbox_min_col:bbox_max_col].tolist().count([0, 0, 0])
        bg_pixel_ratio = bg_pixel_count / (bbox_max_col - bbox_min_col)

        if bg_pixel_ratio > bg_thres:
            break
        else:
            max_row1 = row

    # left boundary move left
    for i in range(1, max_expand_threshold * h_factor):
        col = bbox_min_col - i
        if col<0:
            break

        # Only expand ROI if it doesn't contain too many background pixels
        bg_pixel_count = aligned_img[min_row1:max_row1, col].tolist().count([0, 0, 0])
        bg_pixel_ratio = bg_pixel_count / (max_row1 - min_row1)

        if bg_pixel_ratio > bg_thres:
            break
        else:
            min_col1 = col

    # right boundary move right
    for i in range(1, max_expand_threshold * h_factor):
        col = bbox_max_col + i
        if col>=aligned_img.shape[1]:
            break

        # Only expand ROI if it doesn't contain too many background pixels
        bg_pixel_count = aligned_img[min_row1:max_row1, col].tolist().count([0, 0, 0])
        bg_pixel_ratio = bg_pixel_count / (max_row1 - min_row1)

        if bg_pixel_ratio > bg_thres:
            break
        else:
            max_col1 = col

    area1 = (max_col1-min_col1)*(max_row1-min_row1)

    ''' 2.  try to move left and right boundary first'''
    # left boundary move left
    for i in range(1, max_expand_threshold * h_factor):
        col = bbox_min_col - i
        if col < 0:
            break

        # Only expand ROI if it doesn't contain too many background pixels
        bg_pixel_count = aligned_img[bbox_min_row:bbox_max_row, col].tolist().count([0, 0, 0])
        bg_pixel_ratio = bg_pixel_count / (bbox_max_row - bbox_min_row)

        if bg_pixel_ratio > bg_thres:
            break
        else:
            min_col2 = col

    # right boundary move right
    for i in range(1, max_expand_threshold * h_factor):
        col = bbox_max_col + i
        if col >= aligned_img.shape[1]:
            break

        # Only expand ROI if it doesn't contain too many background pixels
        bg_pixel_count = aligned_img[bbox_min_row:bbox_max_row, col].tolist().count([0, 0, 0])
        bg_pixel_ratio = bg_pixel_count / (bbox_max_row - bbox_min_row)

        if bg_pixel_ratio > bg_thres:
            break
        else:
            max_col2 = col

    # upper boundary move up
    for i in range(1, max_expand_threshold * v_factor):
        row = bbox_min_row - i
        if row < 0:
            break

        # Only expand ROI if it doesn't contain too many background pixels
        bg_pixel_count = aligned_img[row, min_col2:max_col2].tolist().count([0, 0, 0])
        bg_pixel_ratio = bg_pixel_count / (max_col2 - min_col2)

        if bg_pixel_ratio > bg_thres:
            break
        else:
            min_row2 = row

    # lower boundary move down
    for i in range(1, max_expand_threshold * v_factor):
        row = bbox_max_row + i
        if row >= aligned_img.shape[0]:
            break

        # Only expand ROI if it doesn't contain too many background pixels
        bg_pixel_count = aligned_img[row, min_col2:max_col2].tolist().count([0, 0, 0])
        bg_pixel_ratio = bg_pixel_count / (max_col2 - min_col2)

        if bg_pixel_ratio > bg_thres:
            break
        else:
            max_row2 = row

    # Return the indices of the largest area
    area2 = (max_col2-min_col2)*(max_row2-min_row2)

    if area1>area2:
        # Account for rare edge case where bbox has accidentally grown too much
        if area1 > 5 * original_area:
            return [bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col]
        else:
            return [min_row1, min_col1, max_row1, max_col1]
    else:
        # Account for rare edge case where bbox has accidentally grown too much
        if area2 > 5 * original_area:
            return [bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col]
        else:
            return [min_row2, min_col2, max_row2, max_col2]


def ConvertRawStitchLine2BBoxRatio(raw_stitch_line, stitched_img, transform, offset_transform, max_expand_threshold):

    new_stitch_line = []
    for pt in raw_stitch_line:
        row = pt[0]
        col = pt[1]
        new_pt = np.matmul(transform, np.array([row, col, 1]))
        new_stitch_line.append([new_pt[0], new_pt[1]])
    for i in range(len(new_stitch_line)):
        new_stitch_line[i][0] += offset_transform[0, 2]
        new_stitch_line[i][1] += offset_transform[1, 2]
    a = np.transpose(new_stitch_line)
    bbox_min_row = np.floor(np.min(a[0])).astype(int)
    bbox_min_col = np.floor(np.min(a[1])).astype(int)
    bbox_max_row = np.ceil(np.max(a[0])).astype(int)
    bbox_max_col = np.ceil(np.max(a[1])).astype(int)

    [new_min_row, new_min_col, new_max_row, new_max_col] = ExpandROI(stitched_img, bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col, max_expand_threshold=max_expand_threshold)

    rows, cols, channels = stitched_img.shape
    new_min_row_ratio = new_min_row/rows
    new_min_col_ratio = new_min_col/cols
    new_max_row_ratio = new_max_row/rows
    new_max_col_ratio = new_max_col/cols

    return [new_min_row_ratio, new_min_col_ratio, new_max_row_ratio, new_max_col_ratio]


def calculatePoseErr(gt_pose, pose):
    err = np.matmul(gt_pose, np.linalg.inv(pose))
    if np.abs(err[0, 0] - 1) < 1e-3:
        err[0, 0] = 1
    if np.abs(err[0, 0] + 1) < 1e-3:
        err[0, 0] = -1
    r_err = np.arccos(err[0, 0]) * 180 / np.pi
    t_err = np.sqrt(err[0, 2] ** 2 + err[1, 2] ** 2)

    return [r_err, t_err]


def FusionImage(src, dst, transform, bg_color=[0, 0, 0]):

    black_bg = [0, 0, 0]
    if bg_color != black_bg:
        src[np.where((src == bg_color).all(axis=2))] = [0, 0, 0]
        dst[np.where((dst == bg_color).all(axis=2))] = [0, 0, 0]

    color_indices = np.where((dst != black_bg).any(axis=2))
    color_pt_num = len(color_indices[0])
    one = np.ones(color_pt_num)

    color_indices = list(color_indices)
    color_indices.append(one)
    color_indices = np.array(color_indices)

    transformed_lin_pts = np.matmul(transform, color_indices)
    # bounding box after transform
    try:
        dst_min_row = np.floor(np.min(transformed_lin_pts[0])).astype(int)
        dst_min_col = np.floor(np.min(transformed_lin_pts[1])).astype(int)
        dst_max_row = np.ceil(np.max(transformed_lin_pts[0])).astype(int)
        dst_max_col = np.ceil(np.max(transformed_lin_pts[1])).astype(int)
    except ValueError:
        return []  # the src or dst image has the same color with background. e.g totally black.

    # global bounding box
    src_color_indices = np.where((src != black_bg).any(axis=2))
    try:
        src_min_row = np.floor(np.min(src_color_indices[0])).astype(int)
        src_min_col = np.floor(np.min(src_color_indices[1])).astype(int)
        src_max_row = np.ceil(np.max(src_color_indices[0])).astype(int)
        src_max_col = np.ceil(np.max(src_color_indices[1])).astype(int)
    except ValueError:
        return []  # the src or dst image has the same color with background. e.g totally black.

    min_row = min(dst_min_row, src_min_row)
    max_row = max(dst_max_row, src_max_row)
    min_col = min(dst_min_col, src_min_col)
    max_col = max(dst_max_col, src_max_col)

    offset_row = -min_row
    offset_col = -min_col

    offset_transform = np.float32([[1, 0, offset_col], [0, 1, offset_row]])
    point_dst_transform = np.matmul(np.array([[1, 0, offset_row], [0, 1, offset_col], [0, 0, 1]]),
                              transform)

    # convert row, col to opencv x,y
    img_dst_transform = np.float32([[point_dst_transform[0, 0], point_dst_transform[1, 0], point_dst_transform[1, 2]],
                                [point_dst_transform[0, 1], point_dst_transform[1, 1], point_dst_transform[0, 2]]])

    src_transformed = cv2.warpAffine(src, offset_transform, (max_col - min_col, max_row - min_row))
    dst_transformed = cv2.warpAffine(dst, img_dst_transform, (max_col - min_col, max_row - min_row))

    # overlap detection
    a = np.all(src_transformed == black_bg, axis=2)
    b = np.all(dst_transformed != black_bg, axis=2)
    c = np.logical_and(a, b)
    c = c.reshape((c.shape[0], c.shape[1]))
    non_overlap_indices = np.where(c)
    if len(np.where(b)[0]) == 0:
        assert False and "no valid pixels in transformed dst image, please check the transform process"
    else:
        overlap_ratio = 1 - len(non_overlap_indices[0]) / len(np.where(b)[0])

    # fusion
    bg_indices = np.where(a)
    src_transformed[bg_indices] = dst_transformed[bg_indices]

    ### DEBUG ###
    # Padding to make square images
    shape_diff = np.max(src_transformed.shape[:2]) - np.min(src_transformed.shape[:2])
    if np.argmax(src_transformed.shape[:2]) == 0:
        src_transformed_pad = np.pad(
            src_transformed,
            [[0, 0], [0, shape_diff], [0, 0]],
            constant_values=0
        )
    else:
        src_transformed_pad = np.pad(
            src_transformed,
            [[0, shape_diff], [0, 0], [0, 0]],
            constant_values=0
        )
    ### DEBUG ###

    offset_transform_matrix = np.float32([[1, 0, offset_row], [0, 1, offset_col], [0, 0, 1]])

    return [src_transformed_pad, point_dst_transform, overlap_ratio, offset_transform_matrix]
