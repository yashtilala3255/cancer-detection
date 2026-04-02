import os
import shutil
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\yeaht\OneDrive\Desktop\Skin-Cancer-Classification-using-Deep-Learning"
DATA_DIR   = os.path.join(BASE_DIR, "Data")
RAW_DIR    = os.path.join(DATA_DIR, "Raw CSV's")
SAMPLE_DIR = os.path.join(DATA_DIR, "Sample Images")
AUG_DIR    = os.path.join(DATA_DIR, "Augmented Sample Images")
TRAIN_DIR  = os.path.join(DATA_DIR, "train_final")

# 7 clean classes — NO Unknown
CLASS_MAP = {
    "MEL":   "Melanoma",
    "NV":    "Melanocytic_Nevus",
    "BCC":   "Basal_Cell_Carcinoma",
    "AKIEC": "Actinic_Keratosis",
    "AK":    "Actinic_Keratosis",       # 2019 uses AK instead of AKIEC
    "BKL":   "Benign_Keratosis",
    "DF":    "Dermatofibroma",
    "VASC":  "Vascular_Lesion",
    "SCC":   "Squamous_Cell_Carcinoma",
}
CLASSES = sorted(set(CLASS_MAP.values()))
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Final Data Organizer — 2018 + 2019 + 2020 Combined")
print("=" * 60)

# ── STEP 1: Build master label dictionary from all CSVs ───────────────────────
label_dict = {}   # image_name -> class_folder

# --- 2018 Training (one-hot encoded, 10015 rows) ---
path_2018 = os.path.join(RAW_DIR, "2018", "Training", "ISIC2018_Task3_Training_GroundTruth.csv")
df2018 = pd.read_csv(path_2018)
cols_2018 = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
added_2018 = 0
for _, row in df2018.iterrows():
    for col in cols_2018:
        if row[col] == 1.0:
            label_dict[row["image"]] = CLASS_MAP[col]
            added_2018 += 1
            break
print(f"\n2018 Training  : {added_2018} labeled images")

# --- 2018 Validation (one-hot encoded, 193 rows) ---
path_2018v = os.path.join(RAW_DIR, "2018", "Validation", "ISIC2018_Task3_Validation_GroundTruth.csv")
df2018v = pd.read_csv(path_2018v)
added_2018v = 0
for _, row in df2018v.iterrows():
    for col in cols_2018:
        if row[col] == 1.0:
            label_dict[row["image"]] = CLASS_MAP[col]
            added_2018v += 1
            break
print(f"2018 Validation: {added_2018v} labeled images")

# --- 2019 Training (one-hot, skip UNK rows) ---
path_2019 = os.path.join(RAW_DIR, "2019", "ISIC_2019_Training_GroundTruth.csv")
df2019 = pd.read_csv(path_2019)
cols_2019 = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
added_2019 = 0
skipped_unk = 0
for _, row in df2019.iterrows():
    assigned = False
    for col in cols_2019:
        if row[col] == 1.0:
            label_dict[row["image"]] = CLASS_MAP[col]
            added_2019 += 1
            assigned = True
            break
    if not assigned:
        skipped_unk += 1   # skip UNK rows
print(f"2019 Training  : {added_2019} labeled  |  {skipped_unk} Unknown skipped")

# --- 2020 Train (diagnosis column, skip unknown/UNK) ---
path_2020 = os.path.join(RAW_DIR, "2020", "train.csv")
df2020 = pd.read_csv(path_2020)
DIAG_MAP_2020 = {
    "melanoma":              "Melanoma",
    "nevus":                 "Melanocytic_Nevus",
    "basal cell carcinoma":  "Basal_Cell_Carcinoma",
    "actinic keratosis":     "Actinic_Keratosis",
    "squamous cell carcinoma":"Squamous_Cell_Carcinoma",
    "benign keratosis":      "Benign_Keratosis",
    "dermatofibroma":        "Dermatofibroma",
    "vascular lesion":       "Vascular_Lesion",
}
added_2020 = 0
skipped_2020 = 0
for _, row in df2020.iterrows():
    diag = str(row.get("diagnosis","")).strip().lower()
    if diag in DIAG_MAP_2020:
        label_dict[row["image_name"]] = DIAG_MAP_2020[diag]
        added_2020 += 1
    else:
        skipped_2020 += 1
print(f"2020 Training  : {added_2020} labeled  |  {skipped_2020} unknown/skipped")

print(f"\nTotal unique labeled images in master dict: {len(label_dict)}")

# ── STEP 2: Create clean class folders ───────────────────────────────────────
print(f"\nCreating class folders in: {TRAIN_DIR}")
for cls in CLASSES:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    print(f"  {cls}")

# ── STEP 3: Index all available images ───────────────────────────────────────
available = {}
for src_dir in [SAMPLE_DIR, AUG_DIR]:
    if os.path.exists(src_dir):
        for fname in os.listdir(src_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                name_no_ext = os.path.splitext(fname)[0]
                available[name_no_ext] = os.path.join(src_dir, fname)

print(f"\nTotal images found across all image folders: {len(available)}")

# ── STEP 4: Copy images into class folders ────────────────────────────────────
print("\nOrganizing images...")
copied   = 0
skipped  = 0
summary  = {cls: 0 for cls in CLASSES}

for img_name, src_path in available.items():
    cls = label_dict.get(img_name)
    if cls is None:
        skipped += 1
        continue
    dst = os.path.join(TRAIN_DIR, cls, os.path.basename(src_path))
    shutil.copy2(src_path, dst)
    summary[cls] += 1
    copied += 1

# ── STEP 5: Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Done! Final Class Distribution:")
print("=" * 60)
total = 0
for cls in sorted(summary.keys()):
    count = summary[cls]
    bar = "█" * min(count * 2, 40)
    print(f"  {cls:<30} : {count:>4}  {bar}")
    total += count

print(f"\n  Total images copied : {total}")
print(f"  Skipped (no label)  : {skipped}")
print(f"  Output folder       : {TRAIN_DIR}")
print("\n  Now run: python train_final.py")