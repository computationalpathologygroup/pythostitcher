import numpy as np
import copy
import tensorflow as tf

from keras import layers
from tensorflow.keras.applications import EfficientNetB0


class Classifier:
    """
    Helper class to build the fragment classifier ensemble model. Although this is not
    technically an ensemble in the sense of multiple models, we use the average
    prediction of 8 versions (flips and rotations) of the same image.
    """

    def __init__(self, weights):

        self.weights_path = weights
        self.batch_size = 8
        self.model_size = 224
        self.num_classes = 4
        
        # Fix to prevent unfettered VRAM consumption
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def build_model(self):
        """ Build the EfficientNet model """

        # Create base model
        self.model = EfficientNetB0(
            include_top=False, 
            input_shape=(self.model_size, self.model_size, 3), 
            weights=None
        )
        
        # Add classification head
        x = layers.GlobalAveragePooling2D(name="avg_pool")(self.model.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5, name="top_dropout")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="pred")(x)
        
        # Build final model
        self.model = tf.keras.Model(self.model.input, outputs, name="EfficientNet_B0")
        self.model.load_weights(self.weights_path)
        self.model.trainable = False

        return

    def transform_image(self, image, rot, flip):
        """ Convenience function to transform an image """

        # Apply transformations to the image
        rot_factor = -int(rot / 90)
        im_rot = np.rot90(image, rot_factor)

        if flip == "hor":
            im_final = np.fliplr(im_rot)
        elif flip == "ver":
            im_final = np.flipud(im_rot)
        elif flip == "both":
            im_final = np.fliplr(im_rot)
            im_final = np.flipud(im_final)
        elif flip == "none":
            im_final = copy.copy(im_rot)

        return im_final

    def transform_label(self, label, rot, flip):
        """
        Convenience function to transform a label

        Legend:
         - 1 = upper right
         - 2 = lower right
         - 3 = lower left
         - 4 = upper left
        """

        # LUT's for label conversion
        hf_dict = {"1": 4, "2": 3, "3": 2, "4": 1}
        vf_dict = {"1": 2, "2": 1, "3": 4, "4": 3}
        all_labels = [1, 2, 3, 4, 1, 2, 3, 4]

        # Transform label due to flipping
        if flip == "hor":
            rot_label = hf_dict[str(label)]
        elif flip == "ver":
            rot_label = vf_dict[str(label)]
        elif flip == "both":
            rot_label = hf_dict[str(label)]
            rot_label = vf_dict[str(rot_label)]
        elif flip == "none":
            rot_label = copy.copy(label)

        # Transform label due to rotation
        rot_factor = int(rot / 90)
        final_label = all_labels[rot_label - rot_factor - 1]

        return final_label
