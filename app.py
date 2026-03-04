import os
import json
import time
import numpy as np
import tensorflow as tf
import cv2

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage

# =========================================================
# PATH CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "final_effnet_cbam_best.h5")
DISEASE_DB_PATH = os.path.join(BASE_DIR, "disease_db.json")
TRAIN_DIR = os.path.join(BASE_DIR, "plantvillage_split", "train")
GRADCAM_DIR = os.path.join(BASE_DIR, "gradcam_outputs")

IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "top_conv"

os.makedirs(GRADCAM_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL & DATABASE
# =========================================================
print("🔄 Loading model...")
model = load_model(MODEL_PATH, compile=False)
model.trainable = False
print("✅ Model loaded")

with open(DISEASE_DB_PATH, "r") as f:
    DISEASE_DB = json.load(f)

# Class names = folder names (EXACT MATCH - NO NORMALIZATION)
CLASS_NAMES = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])

print("✅ Loaded classes:", CLASS_NAMES)
print("✅ Sample DB keys:", list(DISEASE_DB.keys())[:5])  # Debug: show DB format

# =========================================================
# FLASK APP
# =========================================================
app = Flask(
    __name__,
    static_folder="gradcam_outputs",
    static_url_path="/gradcam_outputs"
)

# =========================================================
# HELPER FUNCTIONS (DIRECT LOOKUP - NO NORMALIZATION)
# =========================================================
def preprocess_image(img_path):
    img = kimage.load_img(img_path, target_size=IMG_SIZE)
    arr = kimage.img_to_array(img)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return img, arr

def generate_gradcam(img_tensor, pred_index):
    conv_layer = model.get_layer(LAST_CONV_LAYER)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def get_care_info(pred_class):
    """DIRECT LOOKUP - EXACT MATCH REQUIRED"""
    care_info = DISEASE_DB.get(pred_class, None)
    
    if care_info is None:
        print(f"❌ No DB match for exact class: '{pred_class}'")
        print(f"   Available DB keys sample: {list(DISEASE_DB.keys())[:3]}...")
    else:
        print(f"✅ Found care info for '{pred_class}'")
    
    return care_info

def is_healthy_class(class_name):
    return class_name.lower().endswith("_healthy")

# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    input_path = os.path.join(GRADCAM_DIR, "input.jpg")
    file.save(input_path)

    # -------- PREPROCESS --------
    orig_img, img_tensor = preprocess_image(input_path)

    # -------- PREDICTION --------
    preds = model.predict(img_tensor)
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx]) * 100
    pred_class = CLASS_NAMES[idx]  # EXACT FOLDER NAME MATCH

    print(f"🧠 Prediction: {pred_class} ({confidence:.2f}%)")

    response = {
        "class": pred_class,
        "confidence": confidence,
        "is_healthy": is_healthy_class(pred_class)
    }

    # -------- HEALTHY CASE --------
    if response["is_healthy"]:
        response["message"] = "The plant is healthy. No care recommendations are required."
        response["care"] = get_care_info(pred_class)  # Direct lookup
        return jsonify(response)

    # -------- GRADCAM --------
    heatmap = generate_gradcam(img_tensor, idx)
    heatmap = cv2.resize(heatmap, orig_img.size)
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(np.array(orig_img), 0.6, heatmap, 0.4, 0)

    timestamp = int(time.time() * 1000)
    gradcam_filename = f"gradcam_{timestamp}.png"
    gradcam_path = os.path.join(GRADCAM_DIR, gradcam_filename)

    cv2.imwrite(
        gradcam_path,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    # -------- CARE INFO (EXACT MATCH) --------
    care_info = get_care_info(pred_class)

    response["gradcam_url"] = f"/gradcam_outputs/{gradcam_filename}"
    response["care"] = care_info
    response["message"] = "Disease detected. Check care recommendations below."

    return jsonify(response)

# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
