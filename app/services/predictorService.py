import os
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import (  # type: ignore
    preprocess_input,
)  # ใช้ preprocess ของ MobileNetV2

# บังคับให้ TensorFlow ใช้ CPU
tf.config.set_visible_devices([], "GPU")

# Define the base path to the models and labels
base_path = os.path.join(os.path.dirname(__file__), "..", "models")

# Define the path to the model file and label file
model_path = os.path.join(base_path, "best_model.h5")
label_path = os.path.join(base_path, "label.txt")

# Load the model
model = tf.keras.models.load_model(model_path)

# Load class names from label.txt with UTF-8 encoding
with open(label_path, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# Define Top-K predictions and confidence threshold
TOP_K = 5  # เลือก top-5 ผลลัพธ์ที่มีความเชื่อมั่นสูงสุด
confidence_threshold = 0.75


def process_and_predict(image: Image.Image):
    try:
        # Resize and preprocess the image for MobileNetV2
        img_resized = image.resize((224, 224))  # Resize to MobileNetV2 input size
        img_array = np.array(img_resized)  # Convert image to array
        img_array = preprocess_input(img_array)  # ใช้ preprocess_input ของ MobileNetV2
        img_array = np.expand_dims(
            img_array, axis=0
        )  # Expand dimensions for batch size

        # Predict using the model
        predictions = model.predict(img_array)[0]  # Get the raw predictions

        # Check if the last layer of the model has a softmax activation
        last_layer = model.layers[-1]
        if (
            isinstance(last_layer, tf.keras.layers.Dense)
            and last_layer.activation != tf.keras.activations.softmax
        ):
            # Apply softmax to convert logits to probabilities if the model doesn't already have softmax
            predictions = tf.nn.softmax(predictions).numpy()

        # Get top-K predictions with confidence scores
        top_k_indices = np.argsort(predictions)[-TOP_K:][::-1]  # Top-K indices
        top_k_predictions = [
            {
                "wound_type": class_names[i],
                "confidence": round(predictions[i] * 100, 2),
            }
            for i in top_k_indices
            if predictions[i] >= confidence_threshold
        ]

        # If no predictions above the threshold, add the default result "ไม่พบแผล"
        if not top_k_predictions:
            top_k_predictions.append({"wound_type": "ไม่พบแผล", "confidence": 0})

        # Convert modified image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        return top_k_predictions, img_byte_arr

    finally:
        # Clean up memory
        del img_resized, img_array, predictions
