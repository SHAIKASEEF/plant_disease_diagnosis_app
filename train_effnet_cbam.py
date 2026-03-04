import os
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from cbam import cbam_block

# ================= GPU SAFETY (IMPORTANT FOR LAPTOP) =================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ================= PATHS =================
TRAIN_DIR = r"C:\Users\Adil\OneDrive\Desktop\plant_disease_diagnosis\plantvillage_split\train"
TEST_DIR  = r"C:\Users\Adil\OneDrive\Desktop\plant_disease_diagnosis\plantvillage_split\test"

CHECKPOINT_DIR = "saved_epochs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================= PARAMETERS =================
IMG_SIZE = 224
BATCH_SIZE = 32
TOTAL_EPOCHS = 20
LR = 1e-4

# ================= FIND LAST SAVED EPOCH =================
def get_last_checkpoint():
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".h5")]
    if not files:
        return None, 0

    epochs = [
        int(re.search(r"epoch_(\d+).h5", f).group(1))
        for f in files
    ]
    last_epoch = max(epochs)
    last_model = os.path.join(
        CHECKPOINT_DIR, f"epoch_{last_epoch:02d}.h5"
    )
    return last_model, last_epoch


last_model_path, last_epoch = get_last_checkpoint()

# ================= DATA =================
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
print("Detected classes:", train_data.class_indices)

# ================= LOAD OR BUILD MODEL =================
if last_model_path:
    print(f"🔁 Resuming from epoch {last_epoch}")
    model = load_model(
        last_model_path,
        custom_objects={"cbam_block": cbam_block}
    )
else:
    print("🚀 Starting training from scratch")

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

model.summary()

# ================= CALLBACKS =================
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "epoch_{epoch:02d}.h5"),
    save_weights_only=False,
    save_freq="epoch",
    verbose=1
)

early_stop_cb = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ================= TRAIN =================
model.fit(
    train_data,
    validation_data=test_data,
    epochs=TOTAL_EPOCHS,
    initial_epoch=last_epoch,
    callbacks=[checkpoint_cb, early_stop_cb]
)
