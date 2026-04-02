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
BASE_DIR  = r"C:\Users\yeaht\OneDrive\Desktop\Skin-Cancer-Classification-using-Deep-Learning"
DATA_DIR  = os.path.join(BASE_DIR, "Data")
TRAIN_DIR = os.path.join(DATA_DIR, "train9")
IMG_SIZE  = 224
BATCH     = 16
EPOCHS    = 30
CLASSES   = 9
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  9-Class Skin Cancer Training")
print("=" * 60)

# ── Check class distribution ──────────────────────────────────────────────────
print("\nClass distribution in train9/:")
class_counts = {}
for cls in sorted(os.listdir(TRAIN_DIR)):
    path = os.path.join(TRAIN_DIR, cls)
    if os.path.isdir(path):
        count = len(os.listdir(path))
        class_counts[cls] = count
        print(f"  {cls:<35} : {count}")

total = sum(class_counts.values())
print(f"\n  Total images: {total}")

# ── Class weights (handle imbalance) ─────────────────────────────────────────
class_weight = {}
classes_sorted = sorted(class_counts.keys())
for i, cls in enumerate(classes_sorted):
    count = class_counts[cls]
    if count > 0:
        class_weight[i] = total / (len(class_counts) * count)
    else:
        class_weight[i] = 1.0
print(f"\nClass weights computed for {len(class_weight)} classes")

# ── Data generators ───────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

print("\nLoading images...")
train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print(f"\nClass indices: {train_gen.class_indices}")
print(f"Train samples : {train_gen.samples}")
print(f"Val samples   : {val_gen.samples}")
num_classes = len(train_gen.class_indices)

# ── Build model ───────────────────────────────────────────────────────────────
print(f"\nBuilding MobileNetV2 model ({num_classes} classes)...")
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_model9.h5", monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=4, min_lr=1e-7, verbose=1)
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

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Training Complete!")
print("=" * 60)
val_loss, val_acc, val_auc = model.evaluate(val_gen, verbose=0)
print(f"  Validation Accuracy : {val_acc*100:.2f}%")
print(f"  Validation AUC      : {val_auc:.4f}")
print(f"  Model saved         : best_model9.h5")

# Save class indices for app.py
import json
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print(f"  Class indices saved : class_indices.json")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig("training_results9.png", dpi=150)
plt.show()
print("  Graph saved: training_results9.png")
print("\n  Update app.py: change MODEL_PATH to 'best_model9.h5'")
print("  Then run: python app.py")