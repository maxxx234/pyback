# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load your trained Keras model
# model = load_model("leaf_disease_model.h5")
# print("Model loaded successfully!")

# # Define your class names
# class_names = ["Bacterial spot", "Blight", "Healthy", "Leaf Spot"]

# # Recommended treatments
# treatments = {
#     "Healthy": "No treatment needed. Maintain healthy practices.",
#     "Bacterial spot": "Remove infected leaves and use copper-based sprays.",
#     "Blight": "Apply fungicides and avoid overhead watering.",
#     "Leaf Spot": "Prune affected areas and use appropriate fungicide."
# }

# @app.route("/detect", methods=["POST"])
# def detect():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files["image"]

#     # Open image and resize
#     img = Image.open(file).convert("RGB").resize((224, 224))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Make prediction
#     pred = model.predict(img_array)[0]
#     index = np.argmax(pred)
#     disease = class_names[index]

#     return jsonify({
#         "disease": disease,
#         "confidence": float(pred[index]),
#         "scores": pred.tolist(),
#         "treatment": treatments[disease]  # ✅ include treatment
#     })

# if __name__ == "__main__":
#     app.run(port=5001, debug=True)


# # https://huggingface.co/sahnkyyyy098/leaf-disease-model/resolve/main/leaf_disease_model.h5from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

from flask_cors import CORS
import requests
import os
from flask import Flask, request, jsonify


import numpy as np





app = Flask(__name__)
CORS(app)

# HuggingFace model URL
MODEL_URL = "https://huggingface.co/sahnkyyyy098/leaf-disease-model/resolve/main/leaf_disease_model.h5"
MODEL_PATH = "leaf_disease_model.h5"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from HuggingFace…")
    response = requests.get(MODEL_URL)

    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    else:
        raise Exception("Failed to download model from HuggingFace")

# Load your trained Keras model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Define class names
class_names = ["Bacterial spot", "Blight", "Healthy", "Leaf Spot"]

# Treatment recommendations
treatments = {
    "Healthy": "No treatment needed. Maintain healthy practices.",
    "Bacterial spot": "Remove infected leaves and use copper-based sprays.",
    "Blight": "Apply fungicides and avoid overhead watering.",
    "Leaf Spot": "Prune affected areas and use appropriate fungicide."
}

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Preprocess image
    img = Image.open(file).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0]
    index = np.argmax(pred)
    disease = class_names[index]

    return jsonify({
        "disease": disease,
        "confidence": float(pred[index]),
        "scores": pred.tolist(),
        "treatment": treatments[disease]
    })

if __name__ == "__main__":
    app.run(port=5001, debug=True)
