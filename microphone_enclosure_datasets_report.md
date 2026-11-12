# Microphone Enclosure Computer-Vision Dataset Report

## 1. Executive Summary

This report documents four image datasets extracted from videos of a physical microphone enclosure and its components:

1. **TOP** — the enclosure cap/top component by itself.
2. **BODY** — the hollow enclosure body by itself.
3. **TOP + BODY (separate)** — cap and body visible side by side, not attached.
4. **COMPLETE ENCLOSURE** — cap and body attached together as an assembled enclosure.

Across the four videos, **1,747 frames were extracted and organized** into training, validation, and test folders. The datasets are structured for YOLO object-detection annotation.

> **Critical status clarification:** These are currently **frame-extracted and annotation-ready datasets—not fully labeled datasets**. Their `labels/` directories were created, but bounding-box annotation files have not yet been populated. Consequently, these datasets cannot be used for supervised YOLO training until annotations are completed and verified.

## 2. Dataset Inventory

| Dataset | Source video | Images | Train | Validation | Test | Detection class(es) |
|---|---|---:|---:|---:|---|
| TOP | `TOP.mp4` | 450 | 315 | 90 | `top` |
| BODY | `BODY.mp4` | 444 | 313 | 86 | `body` |
| TOP + BODY separate | `TOP and BOTTOM.mp4` | 447 | 315 | 90 | `top`, `body` |
| Complete enclosure | `complete-enclosure-covered.mp4` | 406 | 287 | 74 | `complete_enclosure` |
| **Total** | **4 videos** | **1,747** | **1,230** | **340** | — |

*Note: The split counts above are the intended/recorded dataset counts from the individual extraction jobs. The TOP+BODY separate and complete-enclosure test counts should be verified against the actual ZIP contents before training; the extraction logs reported 42 and 45 respectively, while the table's split totals reflect the folder split targets. Use the actual image counts in the ZIP folders as the source of truth.*

## 3. Purpose and Scope

The objective is to train a camera-based model that can locate and identify enclosure components in a live or recorded image:

- Recognize the top/cap when shown independently.
- Recognize the body when shown independently.
- Detect both objects independently when they are side by side and not attached.
- Recognize the assembled object as a complete enclosure when the cap and body are attached.

The selected task is **object detection**, not whole-image classification. Object detection returns a class and bounding box for each visible object, allowing the model to identify the cap and body separately in a single image.

## 4. Dataset Descriptions

### 4.1 TOP Dataset

**Dataset name:** `top_dataset.zip`  
**Source:** `TOP.mp4`  
**Images:** 450  
**Class map (dataset-local):**

| Class ID | Class name |
|---:|---|
| 0 | `top` |

**Purpose:** Train the detector to identify the enclosure cap/top when it is presented as a standalone component.

**Annotation requirement:** Draw one bounding box around the visible top/cap in each image. The matching YOLO label line should use class ID `0`.

### 4.2 BODY Dataset

**Dataset name:** `body_dataset.zip`  
**Source:** `BODY.mp4`  
**Images:** 444  
**Class map (dataset-local):**

| Class ID | Class name |
|---:|---|
| 0 | `body` |

**Purpose:** Train the detector to identify the hollow enclosure body as a standalone component.

**Annotation requirement:** Draw one bounding box around the visible body in each image. The matching YOLO label line should use class ID `0`.

### 4.3 TOP + BODY Separate Dataset

**Dataset name:** `top_body_separate_dataset.zip`  
**Source:** `TOP and BOTTOM.mp4`  
**Images:** 447  
**Class map (dataset-local):**

| Class ID | Class name |
|---:|---|
| 0 | `top` |
| 1 | `body` |

**Purpose:** Teach the detector to identify two distinct objects in the same frame when the cap and body are positioned beside each other but are **not attached**.

**Annotation requirement:** Each image should normally contain two boxes:
- Box around the cap/top: class `0`.
- Box around the body: class `1`.

The two objects must be annotated independently. Do not draw one large box around both separated components.

### 4.4 Complete Enclosure Dataset

**Dataset name:** `complete_enclosure_dataset.zip`  
**Source:** `complete-enclosure-covered.mp4`  
**Images:** 406  
**Class map (dataset-local):**

| Class ID | Class name |
|---:|---|
| 0 | `complete_enclosure` |

**Purpose:** Train the detector to identify the top and body attached together as one assembled enclosure.

**Annotation requirement:** Draw one bounding box around the entire assembled enclosure. Use class ID `0`. Do not label the cap and body as separate objects in these images if the intended target is the assembled enclosure class.

## 5. Frame Extraction and Image Preparation

The videos were processed using OpenCV. The extraction workflow was:

1. Read the source video and its frame count, frame rate, and resolution.
2. Select frame indices evenly across the video, up to approximately 450 candidate frames per video.
3. Read the selected frames.
4. Apply a basic blur-quality filter using the variance of the grayscale Laplacian; frames below the configured threshold were skipped.
5. Save accepted frames as JPEG images at quality setting 95.
6. Place images in train, validation, and test folders based on their position in the sampled sequence.

This approach provides a broad temporal sample of each recording. It does **not** guarantee that all frames are unique, that every angle is equally represented, or that every image is sufficiently sharp and useful for annotation.

## 6. Folder Structure

Each dataset ZIP follows this YOLO-style layout:

```text
<dataset_name>/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
└── README.txt
```

- `images/train/`: images used for model optimization.
- `images/val/`: images used to evaluate model performance during training and tune settings.
- `images/test/`: held-out images used for final evaluation.
- `labels/`: corresponding YOLO annotation text files. These folders are currently empty.
- `data.yaml`: dataset paths and class names.
- `README.txt`: dataset-specific annotation notes.

Every image should have a label file with the same stem. For example:

```text
images/train/top_000123.jpg
labels/train/top_000123.txt
```

## 7. YOLO Annotation Format

Each object is represented by one line in a `.txt` file:

```text
class_id x_center y_center width height
```

The four box values are normalized to the image dimensions and range from 0 to 1:

- `x_center`: horizontal center of the box divided by image width.
- `y_center`: vertical center of the box divided by image height.
- `width`: box width divided by image width.
- `height`: box height divided by image height.

Example with two separated objects:

```text
0 0.30 0.50 0.22 0.35
1 0.70 0.50 0.30 0.55
```

This example is illustrative only; the coordinates must be measured from each actual image. Do not copy these values into the real annotations.

## 8. Recommended Unified Class Map

The four datasets use local class IDs, so **their label IDs must be remapped before combining them**. In particular, class `0` means different things in the TOP, BODY, and COMPLETE datasets.

For one unified model, use this consistent class map:

| Unified ID | Class name | Meaning |
|---:|---|---|
| 0 | `body` | Standalone body or body visible separately |
| 1 | `top` | Standalone cap/top |
| 2 | `complete_enclosure` | Cap and body attached together |

When merging:
- TOP-only annotations: local `0` → unified `1`.
- BODY-only annotations: local `0` → unified `0`.
- TOP+BODY separate annotations: local `0` → unified `1`; local `1` → unified `0`.
- COMPLETE annotations: local `0` → unified `2`.

This unified map enables one model to detect body, top, and assembled enclosure consistently.

## 9. Data Quality and Split Considerations

### 9.1 Video-frame similarity

Frames sampled from a single video can be highly similar. A rotating object may appear nearly unchanged in adjacent sampled frames. This can make the apparent validation/test performance optimistic if similar views are present in training.

### 9.2 Temporal splitting

The extraction process assigned train, validation, and test by chronological position in the sampled frames. This is preferable to randomly distributing adjacent frames, but it is not a fully independent evaluation: the same object, background, lighting, and recording setup may appear in all three splits.

### 9.3 Recommended improvements

Before final training:

- Review frames and remove near-duplicates.
- Remove frames where the object is tiny, occluded, cut off, or too blurred to annotate confidently.
- Keep capture sessions, backgrounds, and lighting conditions separate between train and test where possible.
- Record additional videos with different distances, orientations, backgrounds, and lighting.
- Ensure every split includes meaningful object orientations and sizes.
- Inspect annotations visually after export.

## 10. Annotation Quality Checklist

For every image:

- [ ] The correct object class is selected.
- [ ] The bounding box tightly encloses the visible object.
- [ ] The box does not include unnecessary surrounding background.
- [ ] Both cap and body are independently boxed in TOP+BODY separate images.
- [ ] The assembled enclosure has one complete-object box in COMPLETE images.
- [ ] The annotation file name matches the image file name.
- [ ] Class IDs match the relevant dataset or unified class map.
- [ ] Empty/background-only images have no object annotation lines.
- [ ] Labels are checked for invalid coordinates or missing objects.

## 11. Training Readiness

| Stage | Current status |
|---|---|
| Videos received | Complete |
| Frame extraction | Complete |
| Image folders created | Complete |
| Train/validation/test folders | Created |
| Dataset YAML files | Created |
| Bounding-box annotation | **Pending** |
| Annotation review | Pending |
| Unified dataset merge | Pending |
| YOLO training | Pending |
| Model evaluation | Pending |
| Live webcam deployment | Pending |

The extracted frames alone are not sufficient for supervised object detection. YOLO requires bounding-box labels for the objects in the images.

## 12. Recommended Next Steps

1. **Annotate all four datasets** using CVAT, Roboflow, or another YOLO-compatible tool.
2. Export annotations in YOLO detection format.
3. Verify class IDs and annotation counts.
4. Remap each dataset to the unified class map in Section 8.
5. Merge the images and labels into one dataset, retaining a consistent train/validation/test strategy.
6. Review duplicate frames and split leakage.
7. Train a small YOLO model as a baseline.
8. Evaluate precision, recall, mAP, and per-class errors.
9. Test on new webcam recordings not used during dataset creation.
10. Improve the dataset based on false detections and missed components.

## 13. Conclusion

The four recordings provide a useful initial visual collection for the microphone enclosure detection project. They cover the standalone top, standalone body, separated components, and assembled enclosure states, with **1,747 extracted images in total**.

The dataset organization and class definitions establish a foundation for a unified detector. The most important remaining task is accurate bounding-box annotation, followed by class-ID normalization and careful evaluation on independent images. Once those steps are complete, the data can be used to train and test a YOLO model for live enclosure identification.

---

## Dataset ZIPs

- `top_dataset.zip`
- `body_dataset.zip`
- `top_body_separate_dataset.zip`
- `complete_enclosure_dataset.zip`

**Status:** Frame-extracted and annotation-ready; not yet fully labeled or model-training-ready.
