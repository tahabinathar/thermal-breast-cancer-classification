import os
import glob
import json

import numpy as np
from PIL import Image


IMAGE_SIZE = 224


def get_image_paths(data_root, split, class_name):

    folder = os.path.join(
        data_root,
        split,
        class_name
    )

    paths = glob.glob(
        os.path.join(folder, "*.tif")
    )

    return sorted(paths)


def load_thermal_image(filepath):

    img = Image.open(filepath)

    arr = np.array(
        img,
        dtype=np.float32
    )

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D thermal image, "
            f"got shape {arr.shape}: {filepath}"
        )

    return arr


def calculate_temperature_range(filepaths):

    global_min = np.inf
    global_max = -np.inf

    for filepath in filepaths:

        arr = load_thermal_image(filepath)

        valid = arr > 0

        if not np.any(valid):
            raise ValueError(
                f"No valid temperature pixels found: "
                f"{filepath}"
            )

        current_min = arr[valid].min()
        current_max = arr[valid].max()

        global_min = min(
            global_min,
            current_min
        )

        global_max = max(
            global_max,
            current_max
        )

    if (
        not np.isfinite(global_min)
        or not np.isfinite(global_max)
    ):
        raise ValueError(
            "Invalid temperature range."
        )

    if global_max <= global_min:
        raise ValueError(
            f"Invalid temperature range: "
            f"min={global_min}, "
            f"max={global_max}"
        )

    return (
        float(global_min),
        float(global_max)
    )


def normalize_temperature(
    arr,
    global_min,
    global_max
):

    arr = arr.astype(
        np.float32
    )

    valid = arr > 0

    normalized = np.zeros_like(
        arr,
        dtype=np.float32
    )

    normalized[valid] = (
        (arr[valid] - global_min)
        / (global_max - global_min)
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

    if height > width:

        total_pad = height - width

        pad_left = total_pad // 2
        pad_right = total_pad - pad_left

        return np.pad(
            arr,
            (
                (0, 0),
                (pad_left, pad_right)
            ),
            mode="constant",
            constant_values=0
        )

    else:

        total_pad = width - height

        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top

        return np.pad(
            arr,
            (
                (pad_top, pad_bottom),
                (0, 0)
            ),
            mode="constant",
            constant_values=0
        )


def resize_image(
    arr,
    image_size=IMAGE_SIZE
):

    img = Image.fromarray(
        arr.astype(np.float32)
    )

    img = img.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR
    )

    return np.array(
        img,
        dtype=np.float32
    )


def preprocess_image(
    filepath,
    global_min,
    global_max,
    image_size
):

    arr = load_thermal_image(
        filepath
    )

    if not np.any(arr > 0):
        raise ValueError(
            f"No valid temperature pixels found: "
            f"{filepath}"
        )

    arr = normalize_temperature(
        arr,
        global_min,
        global_max
    )

    arr = pad_to_square(
        arr
    )

    arr = resize_image(
        arr,
        image_size
    )

    arr = arr[..., np.newaxis]

    return arr.astype(
        np.float32
    )


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

            split_counts[split][class_name] = (
                len(paths)
            )

            all_paths.extend(
                paths
            )

    global_min, global_max = (
        calculate_temperature_range(
            all_paths
        )
    )

    failed_files = []

    shape_check = True
    dtype_check = True
    range_check = True
    load_check = True

    expected_shape = (
        IMAGE_SIZE,
        IMAGE_SIZE,
        1
    )

    for filepath in all_paths:

        try:

            arr = load_thermal_image(
                filepath
            )

            if not np.any(arr > 0):
                raise ValueError(
                    "No valid temperature pixels"
                )

            processed = preprocess_image(
                filepath,
                global_min,
                global_max,
                IMAGE_SIZE
            )

            if processed.shape != expected_shape:

                shape_check = False

                raise ValueError(
                    f"Unexpected shape: "
                    f"{processed.shape}"
                )

            if processed.dtype != np.float32:

                dtype_check = False

                raise ValueError(
                    f"Unexpected dtype: "
                    f"{processed.dtype}"
                )

            if (
                np.min(processed) < 0
                or np.max(processed) > 1
            ):

                range_check = False

                raise ValueError(
                    "Processed values outside [0, 1]"
                )

        except Exception as e:

            load_check = False

            failed_files.append({
                "file": filepath,
                "error": str(e)
            })

    total_images = len(
        all_paths
    )

    overall_status = (
        load_check
        and shape_check
        and dtype_check
        and range_check
    )

    results = {

        "total_images": total_images,

        "image_size": IMAGE_SIZE,

        "expected_shape": list(
            expected_shape
        ),

        "expected_dtype": "float32",

        "expected_range": [
            0.0,
            1.0
        ],

        "split_counts": split_counts,

        "temperature_range": {
            "global_min": global_min,
            "global_max": global_max
        },

        "checks": {
            "load_check": load_check,
            "shape_check": shape_check,
            "dtype_check": dtype_check,
            "range_check": range_check
        },

        "failed_files": failed_files,

        "overall_status": overall_status
    }

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        json.dumps(
            results,
            indent=4
        )
    )

    return results


if __name__ == "__main__":

    DATA_ROOT = (
        r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Dataset"
        r"\DMR_IR_Anterior_View_ROI"
    )

    OUTPUT_DIR = (
        r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Experiments"
        r"\EXP_02_DMR_Anterior_ResNet18\Results\Preprocessing_Check"
    )

    os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)
    OUTPUT_PATH = os.path.join(
        OUTPUT_DIR,
        "preprocessing_check.json"
    )

    check_preprocessing(
        DATA_ROOT,
        OUTPUT_PATH
    )