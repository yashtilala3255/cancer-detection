import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── CONFIG ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 20
# UPDATE THIS PATH to your actual Data folder:
DATA_DIR    = r"C:\Users\yeaht\OneDrive\Desktop\Skin-Cancer-Classification-using-Deep-Learning\Data"
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Skin Cancer Detection — Training Script")
print("=" * 60)
print(f"\nTensorFlow version : {tf.__version__}")
print(f"Image size         : {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Max epochs         : {EPOCHS}")
print(f"Data directory     : {DATA_DIR}\n")


# ── STEP 1: Find dataset ─────────────────────────────────────────────────────
# Try to find train/test folders automatically
train_dir = None
test_dir  = None

for root, dirs, files in os.walk(DATA_DIR):
    for d in dirs:
        lower = d.lower()
        if lower == "train":
            train_dir = os.path.join(root, d)
        elif lower in ["test", "val", "valid", "validation"]:
            test_dir = os.path.join(root, d)

if train_dir is None:
    # If no train folder found, use Sample Images folder directly
    sample_dir = os.path.join(DATA_DIR, "Sample Images")
    if os.path.exists(sample_dir):
        print(f"No train/ folder found. Using Sample Images folder: {sample_dir}")
        print("NOTE: For best results, organize images into subfolders per class.")
        print("  e.g. Data/train/benign/  and  Data/train/malignant/\n")
        train_dir = sample_dir
    else:
        print(f"ERROR: Could not find a training folder inside: {DATA_DIR}")
        print("Please make sure your Data folder has this structure:")
        print("  Data/")
        print("    train/")
        print("      benign/      <- benign images here")
        print("      malignant/   <- malignant images here")
        print("    test/")
        print("      benign/")
        print("      malignant/")
        exit(1)

print(f"Train directory : {train_dir}")
print(f"Test  directory : {test_dir if test_dir else 'Not found — will use 20% of train data'}\n")


# ── STEP 2: Data generators ──────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2   # use 20% for validation if no separate test folder
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

print("Loading training images...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

print("Loading validation images...")
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print(f"\nClass labels : {train_generator.class_indices}")
print(f"Training samples   : {train_generator.samples}")
print(f"Validation samples : {val_generator.samples}\n")

if train_generator.samples == 0:
    print("ERROR: No images found!")
    print("Make sure your images are inside class subfolders like:")
    print("  Data/train/benign/image1.jpg")
    print("  Data/train/malignant/image2.jpg")
    exit(1)


# ── STEP 3: Build model (EfficientNetB0 transfer learning) ───────────────────
print("Building EfficientNetB0 model with transfer learning...")

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False   # freeze base layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")   # binary: benign vs malignant
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ── STEP 4: Callbacks ────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]


# ── STEP 5: Train ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Starting Training...")
print("=" * 60 + "\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)


# ── STEP 6: Evaluate ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Training Complete! Evaluating model...")
print("=" * 60 + "\n")

val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"Validation Accuracy : {val_acc * 100:.2f}%")
print(f"Validation Loss     : {val_loss:.4f}")
print(f"\nBest model saved to : best_model.h5")


# ── STEP 7: Plot accuracy and loss curves ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     label="Train Accuracy",  color="blue")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",    color="orange")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["loss"],     label="Train Loss",  color="blue")
axes[1].plot(history.history["val_loss"], label="Val Loss",    color="orange")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("\nTraining graph saved to : training_results.png")


# ── STEP 8: Prediction function ──────────────────────────────────────────────
def predict_image(image_path):
    """Predict a single image — benign or malignant."""
    img   = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr   = img_to_array(img) / 255.0
    arr   = np.expand_dims(arr, axis=0)
    pred  = model.predict(arr, verbose=0)[0][0]
    label = "MALIGNANT" if pred >= 0.5 else "BENIGN"
    conf  = pred if pred >= 0.5 else 1 - pred
    print(f"\nImage   : {os.path.basename(image_path)}")
    print(f"Result  : {label}")
    print(f"Confidence : {conf * 100:.1f}%")
    return label, conf

print("\n" + "=" * 60)
print("  Done! Your model is ready.")
print("=" * 60)
print("\nTo predict a new image, add this at the bottom of the script:")
print('  predict_image(r"path\\to\\your\\image.jpg")')
