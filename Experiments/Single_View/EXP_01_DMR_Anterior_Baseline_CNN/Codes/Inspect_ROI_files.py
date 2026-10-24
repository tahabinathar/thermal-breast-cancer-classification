import os
import glob
import json
import numpy as np
from PIL import Image

# Config
DATA_ROOT = r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Dataset\DMR_IR_Anterior_View_ROI"
RESULTS_DIR = r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Experiments\EXP_01_DMR_Anterior_Baseline_CNN\Results\Inspect_ROI"
SPLITS = ["Train", "Validation", "Test"]
CLASSES = ["Healthy", "Sick"]
THRESHOLD = 50


def get_filepaths(data_root, split, category):
    folder = os.path.join(data_root, split, category)
    filepaths = glob.glob(os.path.join(folder, "*.tif"))
    filepaths = [f for f in filepaths if "mask" not in os.path.basename(f).lower()]
    return filepaths


flagged_files = []
split_counts = {}

for split in SPLITS:
    split_counts[split] = {}

    for category in CLASSES:
        filepaths = get_filepaths(DATA_ROOT, split, category)
        split_counts[split][category] = len(filepaths)

        for filepath in filepaths:
            img = Image.open(filepath)
            arr = np.array(img).astype(np.float32)

            valid = arr[arr > 0]
            if valid.size == 0:
                continue

            unusual = valid[valid > THRESHOLD]
            if unusual.size > 0:
                flagged_files.append({
                    "split": split,
                    "class": category,
                    "file": filepath,
                    "max_value": float(unusual.max()),
                    "unusual_pixels": int(unusual.size),
                    "percentage": float((unusual.size / valid.size) * 100),
                })

# Dataset-wide inspection summary
print("File counts per split/class:")
for split in SPLITS:
    for category in CLASSES:
        print(f"  {split:10s} | {category:8s} | {split_counts[split][category]} files")

print(f"\nFiles flagged (> {THRESHOLD}) across ALL splits: {len(flagged_files)}")
for item in flagged_files:
    print(f"  {item['split']:10s} | {item['class']:8s} | {os.path.basename(item['file'])} | "
          f"max {item['max_value']:.2f} | {item['unusual_pixels']} px ({item['percentage']:.4f}%)")

# Save inspection results
os.makedirs(RESULTS_DIR, exist_ok=True)
output_path = os.path.join(RESULTS_DIR, "dataset_inspection_results.json")

with open(output_path, "w") as f:
    json.dump({
        "threshold": THRESHOLD,
        "split_counts": split_counts,
        "flagged_files": flagged_files,
    }, f, indent=4)

print(f"\nSaved to: {output_path}")