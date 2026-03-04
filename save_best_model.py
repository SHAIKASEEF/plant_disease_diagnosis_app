from tensorflow.keras.models import load_model
from cbam import cbam_block

BEST_EPOCH_PATH = "saved_epochs/epoch_19.h5"
FINAL_MODEL_PATH = "final_effnet_cbam_best.h5"

model = load_model(
    BEST_EPOCH_PATH,
    custom_objects={"cbam_block": cbam_block}
)

model.save(FINAL_MODEL_PATH)
print("✅ Final model saved as:", FINAL_MODEL_PATH)
