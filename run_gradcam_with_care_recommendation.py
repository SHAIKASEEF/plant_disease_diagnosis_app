# run_gradcam_with_care_recommendation.py

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage
import cv2
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
MODEL_PATH = r"final_effnet_cbam_best.h5"

IMG_PATH = r"C:\Users\Adil\OneDrive\Desktop\plant_disease_diagnosis\plantvillage_split\test\Corn_(maize)___Northern_Leaf_Blight\0b112f62-54d5-4395-9f7c-d7ef05290be4___RS_NLB 4196.JPG"
TRAIN_DIR = r"C:\Users\Adil\OneDrive\Desktop\plant_disease_diagnosis\plantvillage_split\train"

DISEASE_DB_PATH = r"C:\Users\Adil\OneDrive\Desktop\plant_disease_diagnosis\disease_db.json"

IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "top_conv"
# ----------------------------------------


# ---------- LOAD CLASS NAMES FROM TRAIN FOLDER ----------
CLASS_NAMES = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])


# ---------- LOAD CARE RECOMMENDATION DATABASE ----------
with open(DISEASE_DB_PATH, "r") as f:
    DISEASE_DB = json.load(f)


def load_and_preprocess(img_path, img_size):
    img = kimage.load_img(img_path, target_size=img_size)
    x = kimage.img_to_array(img)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return img, x


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (orig_img.size[0], orig_img.size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_array = np.array(orig_img)
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def get_care_recommendation(pred_class):
    """
    Converts class name like:
    'Potato___Late_blight' → crop='Potato', disease='Late_blight'
    and fetches info from JSON.
    """
    try:
        crop, disease = pred_class.split("___", 1)
        crop = crop.replace(",", "").strip()
        disease = disease.strip()

        if disease.lower() == "healthy":
            return {
                "message": "The plant is healthy. Continue regular care and monitoring."
            }

        return DISEASE_DB.get(crop, {}).get(disease, None)

    except Exception:
        return None


def main():
    # 1️⃣ Load model
    model = load_model(MODEL_PATH, compile=False)
    model.trainable = False

    # 2️⃣ Load image
    orig_img, img_tensor = load_and_preprocess(IMG_PATH, IMG_SIZE)

    # 3️⃣ Prediction
    preds = model.predict(img_tensor)
    pred_idx = int(np.argmax(preds[0]))
    pred_conf = float(preds[0][pred_idx]) * 100
    pred_class = CLASS_NAMES[pred_idx]

    # 4️⃣ Grad-CAM
    heatmap = make_gradcam_heatmap(
        img_tensor, model, LAST_CONV_LAYER_NAME, pred_index=pred_idx
    )

    overlay = overlay_heatmap_on_image(orig_img, heatmap)

    # 5️⃣ Care recommendation
    care_info = get_care_recommendation(pred_class)

    # 6️⃣ Visualization
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title(f"Original\n{pred_class} ({pred_conf:.2f}%)")
    plt.axis("off")
    plt.imshow(orig_img)

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")
    plt.imshow(overlay)

    plt.tight_layout()

    out_dir = "gradcam_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gradcam_with_care.png")
    plt.savefig(out_path, dpi=200)
    plt.show()

    # 7️⃣ Console output
    print("\n" + "=" * 60)
    print("PLANT DISEASE ANALYSIS REPORT")
    print("=" * 60)
    print(f"Predicted Class : {pred_class}")
    print(f"Confidence      : {pred_conf:.2f}%")

    print("\n🩺 CARE RECOMMENDATION")
    print("-" * 60)

    if care_info:
        for key, value in care_info.items():
            print(f"{key.capitalize():15}: {value}")
    else:
        print("No care recommendation found for this disease.")

    print(f"\nSaved Grad-CAM image at: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
