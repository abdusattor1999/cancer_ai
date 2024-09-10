#Part 1: Metric and Utility Functions
#These functions can be loaded once and used across multiple requests.

import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from io import BytesIO
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the pFBeta metric class
class pFBeta(tf.keras.metrics.Metric):
    """Compute overall probabilistic F-beta score."""
    def __init__(self, beta=1, epsilon=1e-5, name='pF1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.epsilon = epsilon
        self.pos = self.add_weight(name='pos', initializer='zeros')
        self.ctp = self.add_weight(name='ctp', initializer='zeros')
        self.cfp = self.add_weight(name='cfp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        pos = tf.reduce_sum(y_true)
        ctp = tf.reduce_sum(y_pred[y_true == 1])
        cfp = tf.reduce_sum(y_pred[y_true == 0])
        self.pos.assign_add(pos)
        self.ctp.assign_add(ctp)
        self.cfp.assign_add(cfp)

    def result(self):
        beta_squared = self.beta * self.beta
        c_precision = self.ctp / (self.ctp + self.cfp + self.epsilon)
        c_recall = self.ctp / (self.pos + self.epsilon)
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return tf.cond(c_precision > 0 and c_recall > 0, lambda: result, lambda: 0.0)

# Define the threshold-based pFBeta function
def pfbeta_thr(labels, preds):
    thrs = tf.range(0, 1, 0.05)
    best_score = tf.constant(0, dtype=tf.float32)
    for thr in thrs:
        score = labels, tf.cast(preds > thr, tf.float32)
        best_score = tf.cond(score > best_score, lambda: score, lambda: best_score)
    return best_score

pfbeta_thr.__name__ = 'pF1_thr'

# Define custom objects for model loading
custom_objects = {
    "Addons>SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy(),
    "pFBeta": pFBeta,
    "pF1_thr": pfbeta_thr
}

# Function to find the last convolutional layer in the model
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model.")



#####   Part 2: Model Loading
####    This part loads the model with the necessary custom objects. This step can be done once during server initialization.


# Load the pre-trained model
model_path = r'{0}/fold-2.h5'.format(CURRENT_DIR)
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(model_path)




###    Part 3: Grad-CAM Heatmap Generation
###    This part generates the Grad-CAM heatmap for a given image. The function can be called for each request with a new image.



import cv2
import matplotlib.pyplot as plt

# Function to create Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], 
                                       [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to display the heatmap on the image

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    # Encode the processed image in memory
    success, encoded_image = cv2.imencode('.jpg', superimposed_img)
    if not success:
        raise ValueError("Could not encode the image")

    # Return the image as a BytesIO object
    return BytesIO(encoded_image.tobytes())