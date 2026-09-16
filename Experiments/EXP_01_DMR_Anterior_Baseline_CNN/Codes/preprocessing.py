import os
import glob
import json

import numpy as np
from PIL import Image


IMAGE_SIZE = 128


def get_image_paths(data_root, split, class_name):
    folder = os.path.join(
        data_root,
        split,
        class_name
    )

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Directory not found: {folder}"
        )

    paths = glob.glob(
        os.path.join(folder, "*.tif")
    )

    if not paths:
        raise FileNotFoundError(
            f"No TIFF images found: {folder}"
        )

    return sorted(paths)


def load_thermal_image(filepath):
    with Image.open(filepath) as img:
        arr = np.array(img, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D thermal image, "
            f"got shape {arr.shape}: {filepath}"
        )

    return arr


def calculate_global_temperature_range(filepaths):
    global_min = None
    global_max = None

    for filepath in filepaths:
        arr = load_thermal_image(filepath)
        valid = arr[arr > 0]

        if valid.size == 0:
            continue

        file_min = float(valid.min())
        file_max = float(valid.max())

        if global_min is None or file_min < global_min:
            global_min = file_min

        if global_max is None or file_max > global_max:
            global_max = file_max

    if global_min is None or global_max is None:
        raise ValueError(
            "No valid temperature values found "
            "across provided filepaths."
        )

    return global_min, global_max


def normalize_temperature(arr, global_min, global_max):
    if global_max <= global_min:
        raise ValueError(
            "global_max must be greater than global_min."
        )

    arr = np.asarray(arr, dtype=np.float32)

    normalized = np.zeros_like(
        arr,
        dtype=np.float32
    )

    valid = arr > 0

    normalized[valid] = (
        arr[valid] - global_min
    ) / (
        global_max - global_min
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized


def pad_to_square(arr):
    height, width = arr.shape

    if height == width:
        return arr

    size = max(height, width)

    padded = np.zeros(
        (size, size),
        dtype=arr.dtype
    )

    y_offset = (size - height) // 2
    x_offset = (size - width) // 2

    padded[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = arr

    return padded


def resize_image(arr, image_size=IMAGE_SIZE):
    image = Image.fromarray(
        arr.astype(np.float32)
    )

    image = image.resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR
    )

    return np.asarray(
        image,
        dtype=np.float32
    )


def preprocess_image(
    filepath,
    global_min,
    global_max,
    image_size=IMAGE_SIZE
):
    arr = load_thermal_image(filepath)

    if not np.any(arr > 0):
        raise ValueError(
            f"No valid temperature values found: {filepath}"
        )

    arr = normalize_temperature(
        arr,
        global_min,
        global_max
    )

    arr = pad_to_square(arr)

    arr = resize_image(
        arr,
        image_size
    )

    arr = np.expand_dims(
        arr,
        axis=-1
    )

    return arr.astype(np.float32)


def check_preprocessing(
    data_root,
    output_path
):
    splits = [
        "Train",
        "Validation",
        "Test"
    ]

    classes = [
        "Healthy",
        "Sick"
    ]

    all_paths = []
    split_counts = {}

    for split in splits:
        split_counts[split] = {}

        for class_name in classes:
            paths = get_image_paths(
                data_root,
                split,
                class_name
            )

            split_counts[split][class_name] = len(paths)
            all_paths.extend(paths)

    # Calculate the temperature range
    global_min, global_max = (
        calculate_global_temperature_range(
            all_paths
        )
    )

    checks = {
        "images_loaded": True,
        "images_are_2d": True,
        "valid_temperature_values": True,
        "temperature_range_calculated": True,
        "normalization_range": True,
        "output_shape": True,
        "output_dtype": True
    }

    failed_files = []

    for filepath in all_paths:
        try:
            raw = load_thermal_image(filepath)

            if raw.ndim != 2:
                raise ValueError(
                    f"Expected 2D image, got {raw.shape}"
                )

            if not np.any(raw > 0):
                raise ValueError(
                    "No valid temperature values"
                )

            processed = preprocess_image(
                filepath,
                global_min,
                global_max,
                IMAGE_SIZE
            )

            if not np.all(
                (processed >= 0.0) &
                (processed <= 1.0)
            ):
                raise ValueError(
                    "Output contains values outside [0, 1]"
                )

            if processed.shape != (
                IMAGE_SIZE,
                IMAGE_SIZE,
                1
            ):
                raise ValueError(
                    f"Unexpected output shape: "
                    f"{processed.shape}"
                )

            if processed.dtype != np.float32:
                raise ValueError(
                    f"Unexpected output dtype: "
                    f"{processed.dtype}"
                )

        except Exception as exc:
            failed_files.append({
                "file": filepath,
                "error": str(exc)
            })

    overall_status = (
        "passed"
        if len(failed_files) == 0
        else "failed"
    )

    result = {
        "total_images_checked": len(all_paths),
        "image_size": IMAGE_SIZE,
        "expected_output_shape": [
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        ],
        "expected_dtype": "float32",
        "expected_value_range": [
            0.0,
            1.0
        ],
        "split_counts": split_counts,
        "temperature_range": {
            "global_min": global_min,
            "global_max": global_max
        },
        "checks": checks,
        "failed_files": failed_files,
        "overall_status": overall_status
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=4
        )

    return result


if __name__ == "__main__":

    DATA_ROOT = r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Dataset\DMR_IR_Anterior_View_ROI"

    OUTPUT_DIR = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Experiments\EXP_01_DMR_Anterior_Baseline_CNN"
    r"\Results\Preprocessing_Check"
    )

    OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "preprocessing_check.json"
    )

    result = check_preprocessing(
    data_root=DATA_ROOT,
    output_path=OUTPUT_PATH
    )


    print(
        f"Preprocessing check: "
        f"{result['overall_status']}"
    )

    print(
        f"Images checked: "
        f"{result['total_images_checked']}"
    )

    print(
        f"Results saved to: "
        f"{OUTPUT_PATH}"
    )