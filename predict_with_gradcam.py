import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D

# ===================== PATHS =====================
MODEL_PATH = r"baseline_cnn_plantvillage.h5"

IMAGE_PATH = r"C:\Users\Adil\OneDrive\Desktop\plant_disease_diagnosis\plantvillage_split\test\Blueberry___healthy\0a818f22-929b-4ef8-bcdb-ac86e909ba26___RS_HL 5438_final_masked.jpg"

TRAIN_DIR = r"C:\Users\Adil\OneDrive\Desktop\plant_disease_diagnosis\plantvillage_split\train"

IMG_SIZE = 224

# ===================== LOAD MODEL =====================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Remove softmax (CRITICAL)
if hasattr(model.layers[-1], "activation"):
    model.layers[-1].activation = tf.keras.activations.linear

logit_model = tf.keras.models.clone_model(model)
logit_model.set_weights(model.get_weights())

# ===================== CLASS LABELS =====================
class_labels = sorted(os.listdir(TRAIN_DIR))

# ===================== LOAD IMAGE =====================
img = image.load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_norm = img_array / 255.0
input_tensor = np.expand_dims(img_norm, axis=0)
# ensure input is a tf.Tensor so GradientTape records ops
input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

# ===================== PREDICTION =====================
logits = logit_model(input_tensor, training=False)
preds = tf.nn.softmax(logits).numpy()
pred_index = np.argmax(preds)
confidence = preds[0][pred_index] * 100
predicted_class = class_labels[pred_index]

print(f"\n🌿 Predicted Class: {predicted_class}")
print(f"📊 Confidence: {confidence:.2f}%")

# ===================== FIND LAST CONV =====================
last_conv_layer = None
for layer in reversed(logit_model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise RuntimeError("No Conv2D layer found")

print(f"✅ Last Conv Layer: {last_conv_layer.name}")

# ===================== GRAD MODEL =====================
grad_model = tf.keras.models.Model(
    inputs=logit_model.inputs,
    outputs=[last_conv_layer.output, logit_model.outputs[0]]
)

# ===================== GRAD-CAM++ =====================
with tf.GradientTape(persistent=True) as tape:
    conv_outputs, logits = grad_model(input_tensor, training=False)
    # ensure conv_outputs is watched for higher-order gradients
    tape.watch(conv_outputs)
    score = logits[:, pred_index]

# First-order gradients (Grad-CAM). Attempt higher-order for Grad-CAM++,
# but fall back to standard Grad-CAM if second/third derivatives are unavailable.
grads = tape.gradient(score, conv_outputs)

grads2 = None
grads3 = None
try:
    grads2 = tape.gradient(grads, conv_outputs)
    grads3 = tape.gradient(grads2, conv_outputs)
except Exception:
    grads2 = None
    grads3 = None

if grads is None:
    raise RuntimeError("Gradients are None; cannot compute Grad-CAM")

conv_outputs = conv_outputs[0]
grads = grads[0]

if grads2 is None or grads3 is None:
    # Fallback: standard Grad-CAM
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    weights = pooled_grads
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
else:
    grads2 = grads2[0]
    grads3 = grads3[0]
    # Grad-CAM++ weights
    numerator = grads2
    denominator = 2 * grads2 + grads3 * conv_outputs
    denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))

    alpha = numerator / denominator
    weights = tf.reduce_sum(alpha * tf.maximum(grads, 0), axis=(0, 1))

    # Weighted sum
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)

# Normalize
heatmap = tf.maximum(heatmap, 0)
heatmap = heatmap / tf.reduce_max(heatmap)
heatmap = heatmap.numpy()

# ===================== POST-PROCESSING =====================
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

# Remove weak noise
heatmap[heatmap < 0.3] = 0

# Smooth
heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

# Apply JET
heatmap_uint8 = np.uint8(255 * heatmap)
heatmap_jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

# ===================== OVERLAY =====================
original_img = img_array.astype(np.uint8)

alpha_overlay = 0.7
overlay = cv2.addWeighted(
    original_img, 1 - alpha_overlay,
    heatmap_jet, alpha_overlay,
    0
)

# ===================== DISPLAY =====================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_norm)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Grad-CAM++ (JET Heatmap)")
plt.axis("off")

plt.tight_layout()
plt.show()
