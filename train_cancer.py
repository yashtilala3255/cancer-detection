import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\yeaht\OneDrive\Desktop\Skin-Cancer-Classification-using-Deep-Learning"
DATA_DIR   = os.path.join(BASE_DIR, "Data")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 25
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Skin Cancer Detection — Final Training Script")
print("=" * 60)

benign_dir    = os.path.join(TRAIN_DIR, "benign")
malignant_dir = os.path.join(TRAIN_DIR, "malignant")
benign_count    = len(os.listdir(benign_dir))
malignant_count = len(os.listdir(malignant_dir))

print(f"\nBenign    : {benign_count}")
print(f"Malignant : {malignant_count}")

total       = benign_count + malignant_count
w_benign    = (1 / benign_count)    * total / 2.0
w_malignant = (1 / malignant_count) * total / 2.0
class_weight = {0: w_benign, 1: w_malignant}
print(f"Class weights → benign: {w_benign:.3f}, malignant: {w_malignant:.3f}")

# ── Data generators ───────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)

print("\nLoading images...")
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print(f"Class map    : {train_gen.class_indices}")
print(f"Train images : {train_gen.samples}")
print(f"Val images   : {val_gen.samples}")

# ── Build model using MobileNetV2 (better for small datasets) ─────────────────
print("\nBuilding MobileNetV2 model...")

base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Unfreeze last 30 layers for fine-tuning
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

trainable = sum(1 for l in base.layers if l.trainable)
print(f"Trainable base layers : {trainable} / {len(base.layers)}")

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_auc", patience=6,
                  restore_best_weights=True, mode="max", verbose=1),
    ModelCheckpoint("best_model.h5", monitor="val_auc",
                    save_best_only=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1)
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Starting Training...")
print("=" * 60 + "\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Training Complete!")
print("=" * 60)
val_loss, val_acc, val_auc = model.evaluate(val_gen, verbose=0)
print(f"  Validation Accuracy : {val_acc * 100:.2f}%")
print(f"  Validation AUC      : {val_auc:.4f}")
print(f"  Best model saved    : best_model.h5")

# ── Bias check ────────────────────────────────────────────────────────────────
print("\nChecking predictions on sample images...")
print(f"{'Image':<30} {'True Label':<12} {'Predicted':<12} {'Confidence'}")
print("-" * 68)

def predict_one(path, true_label):
    img  = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr  = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    pred = float(model.predict(arr, verbose=0)[0][0])
    label = "MALIGNANT" if pred >= 0.5 else "BENIGN"
    conf  = pred if pred >= 0.5 else 1 - pred
    match = "OK" if label.lower() == true_label.lower() else "WRONG"
    print(f"{os.path.basename(path)[:28]:<30} {true_label:<12} {label:<12} {conf*100:.1f}%  {match}")

for f in os.listdir(benign_dir)[:5]:
    predict_one(os.path.join(benign_dir, f), "benign")

orig_malignant = [f for f in os.listdir(malignant_dir) if not f.startswith("aug_")]
for f in orig_malignant[:5]:
    predict_one(os.path.join(malignant_dir, f), "malignant")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)

axes[2].plot(history.history["auc"],     label="Train")
axes[2].plot(history.history["val_auc"], label="Val")
axes[2].set_title("AUC Score"); axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("\nGraph saved : training_results.png")
print("Now run    : python app.py")