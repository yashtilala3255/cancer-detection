import os
import shutil
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR      = r"C:\Users\yeaht\OneDrive\Desktop\Skin-Cancer-Classification-using-Deep-Learning"
DATA_DIR      = os.path.join(BASE_DIR, "Data")
IMAGES_DIR    = os.path.join(DATA_DIR, "Sample Images")
CSV_FILE      = os.path.join(DATA_DIR, "Raw CSV's", "2020", "train.csv")
OUTPUT_DIR    = os.path.join(DATA_DIR, "train")
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Data Organizer — Skin Cancer Project")
print("=" * 60)
print(f"\nUsing CSV     : {CSV_FILE}")
print(f"Images folder : {IMAGES_DIR}")
print(f"Output folder : {OUTPUT_DIR}\n")

# ── STEP 1: Check CSV exists ──────────────────────────────────────────────────
if not os.path.exists(CSV_FILE):
    print(f"ERROR: train.csv not found at:\n  {CSV_FILE}")
    print("\nAvailable files in Raw CSV's/2020:")
    folder = os.path.join(DATA_DIR, "Raw CSV's", "2020")
    for f in os.listdir(folder):
        print(f"  {f}")
    exit(1)

# ── STEP 2: Load CSV ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
print(f"Loaded train.csv successfully!")
print(f"Total rows : {len(df)}")
print(f"Columns    : {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nLabel distribution:")
print(df["target"].value_counts())

# ── STEP 3: Create output folders ────────────────────────────────────────────
benign_dir    = os.path.join(OUTPUT_DIR, "benign")
malignant_dir = os.path.join(OUTPUT_DIR, "malignant")
os.makedirs(benign_dir,    exist_ok=True)
os.makedirs(malignant_dir, exist_ok=True)
print(f"\nFolders created:")
print(f"  {benign_dir}")
print(f"  {malignant_dir}")

# ── STEP 4: List all available images ────────────────────────────────────────
available = {}
for fname in os.listdir(IMAGES_DIR):
    name_no_ext = os.path.splitext(fname)[0]
    available[name_no_ext] = os.path.join(IMAGES_DIR, fname)

print(f"\nTotal images found in Sample Images folder: {len(available)}")

# ── STEP 5: Copy images ───────────────────────────────────────────────────────
print("\nOrganizing images...")
copied    = 0
not_found = 0

for _, row in df.iterrows():
    image_name = str(row["image_name"]).strip()
    label      = int(row["target"])

    if image_name in available:
        src  = available[image_name]
        dest = malignant_dir if label == 1 else benign_dir
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
        copied += 1
    else:
        not_found += 1

# ── STEP 6: Summary ───────────────────────────────────────────────────────────
benign_count    = len(os.listdir(benign_dir))
malignant_count = len(os.listdir(malignant_dir))

print("\n" + "=" * 60)
print("  Done! Summary:")
print("=" * 60)
print(f"  Benign images    : {benign_count}")
print(f"  Malignant images : {malignant_count}")
print(f"  Total copied     : {copied}")
print(f"  Not found in CSV : {not_found}")

if copied == 0:
    print("\n  WARNING: 0 images copied!")
    print("  This means your Sample Images filenames don't match train.csv.")
    print("\n  Sample Images filenames look like:")
    sample_files = list(os.listdir(IMAGES_DIR))[:5]
    for f in sample_files:
        print(f"    {f}")
    print("\n  train.csv image_name column looks like:")
    for name in df["image_name"].head(5):
        print(f"    {name}")
else:
    print(f"\n  Now run: python train_cancer.py")