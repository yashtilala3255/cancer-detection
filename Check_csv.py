import pandas as pd
import os

BASE = r"C:\Users\yeaht\OneDrive\Desktop\Skin-Cancer-Classification-using-Deep-Learning\Data"

files = {
    "2018 Training GT":   os.path.join(BASE, "Raw CSV's", "2018", "Training", "ISIC2018_Task3_Training_GroundTruth.csv"),
    "2018 Validation GT": os.path.join(BASE, "Raw CSV's", "2018", "Validation", "ISIC2018_Task3_Validation_GroundTruth.csv"),
    "2019 Training GT":   os.path.join(BASE, "Raw CSV's", "2019", "ISIC_2019_Training_GroundTruth.csv"),
    "2020 Train":         os.path.join(BASE, "Raw CSV's", "2020", "train.csv"),
    "Processed 9Labels":  os.path.join(BASE, "Processed CSV's", "train_2020_and_2019_with_9_Labels.csv"),
    "Processed 4Labels":  os.path.join(BASE, "Processed CSV's", "train_2020_and_2019_with_4_Labels.csv"),
    "2019 Metadata":      os.path.join(BASE, "Raw CSV's", "2019", "Patient MetaData", "ISIC_2019_Training_Metadata.csv"),
}

for name, path in files.items():
    print("=" * 60)
    print(f"FILE: {name}")
    if not os.path.exists(path):
        print("  NOT FOUND")
        continue
    df = pd.read_csv(path)
    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Sample  :\n{df.head(3).to_string()}")
    print()