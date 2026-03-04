import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from cbam import cbam_block

# ================= PATHS =================
TEST_DIR = r"C:\Users\Adil\OneDrive\Desktop\plant_disease_diagnosis\plantvillage_split\test"
EPOCHS_DIR = r"C:\Users\Adil\OneDrive\Desktop\plant_disease_diagnosis\saved_epochs"

IMG_SIZE = 224
BATCH_SIZE = 32

# ================= TEST DATA =================
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ================= EVALUATE EACH EPOCH =================
results = []

epoch_files = sorted(
    [f for f in os.listdir(EPOCHS_DIR) if f.endswith(".h5")]
)

print("\n📊 Evaluating saved epochs...\n")

for epoch_file in epoch_files:
    model_path = os.path.join(EPOCHS_DIR, epoch_file)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"cbam_block": cbam_block}
    )

    loss, acc = model.evaluate(test_data, verbose=0)
    results.append((epoch_file, acc))

    print(f"{epoch_file} → Accuracy: {acc:.4f}")

# ================= BEST EPOCH =================
best_epoch = max(results, key=lambda x: x[1])

print("\n🏆 BEST MODEL")
print(f"Epoch file : {best_epoch[0]}")
print(f"Accuracy   : {best_epoch[1]:.4f}")
