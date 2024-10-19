#Part 1: Metric and Utility Functions
#These functions can be loaded once and used across multiple requests.

import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from io import BytesIO
import os

BREAST_MODEL = "breast_model.h5"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

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
model_path = r'{0}/{1}'.format(CURRENT_DIR, BREAST_MODEL)
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


#################################################################################################################################################

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
from fastapi import File, UploadFile
import shutil

router = APIRouter(prefix="/cancer")

UPLOAD_DIR = os.makedirs("upload_images", exist_ok=True)
PROCESSED_IMAGES = Path("processed_images")


CANCER_THRESHOLD_MEASURES = {
    "no": "No cancer detected, everything is ok !",
    "low": "There are some unclear signs, and it's uncertain whether these indicate cancer. Further diagnostic tests or a consultation with a specialist is necessary",
    "medium": "The images show suspicious areas that could indicate the presence of cancer. Please consult with a healthcare provider as soon as possible for a more thorough examination",
    "strong": "The images indicate a high likelihood of cancer. Immediate medical attention is required. Please consult an oncologist or your healthcare provider for the next steps.",
}

def get_description_message(threshold):
    key = "no"
    if 0 < threshold <= 0.2:
        key = "low"
    elif 0.2 < threshold <= 0.4:
        key = "medium"
    else:
        key = "strong"
    return CANCER_THRESHOLD_MEASURES[key]

def test_is_cancer(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(1024, 512))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Normalize the image if required by the model
    img_array = img_array / 255.0  # If the model was trained on normalized data
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    last_conv_layer_name = find_last_conv_layer(model)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    processed_img_io = save_and_display_gradcam(image_path, heatmap)
    predictions = model.predict(img_array)


    return processed_img_io, get_description_message(predictions[0])


@router.post("/breast/")
async def process_photo(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary location
    file_path = f"upload_images/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image using test_is_cancer
    processed_img_io, description = test_is_cancer(file_path)

    # Clean up the uploaded file after processing
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")

    return StreamingResponse(
        content=processed_img_io,
        media_type="image/jpeg",
        headers={"Description": description}
    )
