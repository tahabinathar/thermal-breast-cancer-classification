# EXP02 Preprocessing Check

This check verifies the preprocessing pipeline used for the EXP02 ResNet-18 model on the DMR-IR Anterior View ROI dataset.

## Preprocessing Pipeline

Each thermal image is processed as follows:

1. **Load thermal image**

   The `.tif` image is loaded as a 2D floating-point array.

2. **Calculate temperature range**

   The minimum and maximum valid temperature values (`> 0`) are calculated from all images used for the preprocessing check.

   The calculated global temperature range is:

   * Minimum: `22.0`
   * Maximum: `36.812618255615234`

3. **Temperature normalization**

   Valid temperature values are normalized to the range `[0, 1]` using:

   `normalized = (temperature - global_min) / (global_max - global_min)`

   Invalid/background pixels (`≤ 0`) remain zero.

4. **Zero padding**

   Non-square images are padded with zeros to produce a square image. The original image is centered within the padded array.

5. **Resize**

   The square image is resized to `224 × 224` pixels using bilinear interpolation.

6. **Add channel dimension**

   A single channel dimension is added, producing an output shape of:

   `224 × 224 × 1`

7. **Output format**

   The final array is stored as `float32`.

## Verification

The standalone preprocessing check verifies that:

* all images load successfully;

* all images are 2D;

* valid temperature values are present;

* the temperature range can be calculated;

* normalized values remain within `[0, 1]`;

* the final output shape is `224 × 224 × 1`;

* the final output dtype is `float32`.

## Dataset

The 234 images included in the preprocessing check are distributed as follows:

* **Train:** 163 images

  * Healthy: 103
  * Sick: 60
* **Validation:** 34 images

  * Healthy: 22
  * Sick: 12
* **Test:** 37 images

  * Healthy: 24
  * Sick: 13

## Result

All **234 images** passed the preprocessing checks with no failed files.

The detailed verification results are stored in `preprocessing_check.json`.

> **Note:** This check verifies the preprocessing implementation. The temperature range used during actual model development must be calculated from the appropriate training/development data and then applied to validation and test data separately.
