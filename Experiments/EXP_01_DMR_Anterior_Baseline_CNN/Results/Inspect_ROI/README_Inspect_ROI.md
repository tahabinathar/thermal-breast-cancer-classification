# EXP01 ROI Pixel Value Inspection

This inspection was performed on the cropped **DMR-IR Dataset** regions of interest (ROIs) to identify any abnormal pixel values.

A threshold of **50** was used to flag potentially abnormal pixels across the Train, Validation, and Test splits.

**Result:** No abnormal pixel values were detected in any image (`flagged_files: []`).

| Split      | Healthy |   Sick |
| ---------- | ------: | -----: |
| Train      |     103 |     60 |
| Validation |      22 |     12 |
| Test       |      24 |     13 |
| **Total**  | **149** | **85** |

All **234 ROI images** passed the inspection without flagged pixel values.