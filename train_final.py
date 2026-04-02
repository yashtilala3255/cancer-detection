import os
import json
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
TRAIN_DIR = os.path.join(DATA_DIR, "train_final")
IMG_SIZE  = 224
BATCH     = 16
EPOCHS    = 35
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Final Skin Cancer Training — No Unknown Class")
print("=" * 60)

# ── Class distribution ────────────────────────────────────────────────────────
print("\nClass distribution:")
class_counts = {}
for cls in sorted(os.listdir(TRAIN_DIR)):
    path = os.path.join(TRAIN_DIR, cls)
    if os.path.isdir(path):
        count = len(os.listdir(path))
        class_counts[cls] = count
        print(f"  {cls:<35} : {count}")

total = sum(class_counts.values())
print(f"\n  Total: {total} images across {len(class_counts)} classes")

if total == 0:
    print("\nERROR: No images found! Run organize_data.py first.")
    exit(1)

# ── Class weights ─────────────────────────────────────────────────────────────
classes_sorted = sorted(class_counts.keys())
class_weight = {}
for i, cls in enumerate(classes_sorted):
    count = class_counts[cls]
    class_weight[i] = (total / (len(class_counts) * count)) if count > 0 else 1.0
print(f"\nClass weights: {class_weight}")

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
    shear_range=0.1,
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

num_classes = len(train_gen.class_indices)
print(f"\nClass indices : {train_gen.class_indices}")
print(f"Train samples : {train_gen.samples}")
print(f"Val samples   : {val_gen.samples}")
print(f"Num classes   : {num_classes}")

# ── Save class indices ────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "class_indices.json"), "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print("Class indices saved to class_indices.json")

# ── Build model ───────────────────────────────────────────────────────────────
print(f"\nBuilding MobileNetV2 — {num_classes} classes...")
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

trainable = sum(1 for l in base.layers if l.trainable)
print(f"Trainable base layers: {trainable}/{len(base.layers)}")

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
model_path = os.path.join(BASE_DIR, "best_model_final.h5")
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(model_path, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=4, min_lr=1e-8, verbose=1)
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Training Started...")
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
print(f"  Validation Accuracy : {val_acc*100:.2f}%")
print(f"  Validation AUC      : {val_auc:.4f}")
print(f"  Model saved         : {model_path}")

# ── Quick prediction check ────────────────────────────────────────────────────
print("\nSample predictions (bias check):")
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
print(f"{'Class':<35} {'Predicted':<35} {'Match'}")
print("-" * 80)

for cls_name in sorted(os.listdir(TRAIN_DIR)):
    cls_path = os.path.join(TRAIN_DIR, cls_name)
    if not os.path.isdir(cls_path): continue
    files = os.listdir(cls_path)
    if not files: continue
    fpath = os.path.join(cls_path, files[0])
    img   = load_img(fpath, target_size=(IMG_SIZE, IMG_SIZE))
    arr   = np.expand_dims(img_to_array(img)/255.0, 0)
    pred  = model.predict(arr, verbose=0)[0]
    pred_cls = idx_to_class.get(int(np.argmax(pred)), "?")
    conf  = round(float(np.max(pred))*100, 1)
    match = "✓" if pred_cls == cls_name else "✗"
    print(f"  {cls_name:<33} → {pred_cls:<33} {conf}%  {match}")

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
axes[2].set_title("AUC"); axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "training_results_final.png"), dpi=150)
plt.show()
print("\nGraph saved: training_results_final.png")
print("\nNext steps:")
print("  1. Open app.py")
print("  2. Change MODEL_PATH to 'best_model_final.h5'")
print("  3. Run: python app.py")