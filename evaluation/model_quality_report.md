# Model quality report — run `v2`

## Data used
```
{
  "body": {
    "train": 279,
    "val": 61,
    "test": 61
  },
  "top": {
    "train": 283,
    "val": 62,
    "test": 62
  },
  "top_body_separate": {
    "train": 211,
    "val": 47,
    "test": 47
  },
  "complete_enclosure": {
    "train": 251,
    "val": 55,
    "test": 55
  }
}
```
All labels human-verified (annotations/); unlabeled frames were dropped by user choice. Splits are contiguous time blocks per video with 5-frame gaps.

## Training curves (from results.csv)
- epochs run: 60; best epoch (val mAP50-95): 41
- train box/cls loss at last epoch: 0.806 / 0.299
- val box/cls loss at last epoch: 1.041 / 0.436
- val box/cls loss at best epoch: 0.963 / 0.373
- val min box loss 0.963 at epoch 41
- final val mAP50 0.975, mAP50-95 0.644

## Over/underfitting analysis
- final val/train box-loss ratio: 1.29
- val box loss rose >25% above its minimum late in training: no
- train box loss still high (>1.5) and val mAP50-95 < 0.5 (underfitting sign): no
Note: validation frames come from the same videos as training, so a good val score does not prove generalization.

## Validation (model-selection split)
- precision 0.959, recall 0.973, mAP50 0.986, mAP50-95 0.681

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| body | 0.939 | 1.000 | 0.992 | 0.727 |
| top | 1.000 | 0.973 | 0.995 | 0.628 |
| complete_enclosure | 0.939 | 0.945 | 0.972 | 0.689 |

## Held-out test (final)
- precision 0.996, recall 0.996, mAP50 0.992, mAP50-95 0.809

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| body | 0.994 | 1.000 | 0.995 | 0.789 |
| top | 0.999 | 0.988 | 0.985 | 0.813 |
| complete_enclosure | 0.995 | 1.000 | 0.995 | 0.824 |

## Per-source-dataset test (conf 0.5, IoU 0.5)
| dataset | images | TP | FP | FN | wrong class | precision | recall |
|---|---|---|---|---|---|---|---|
| body | 61 | 61 | 0 | 0 | 0 | 1.000 | 1.000 |
| top | 62 | 62 | 0 | 0 | 0 | 1.000 | 1.000 |
| top_body_separate | 47 | 68 | 0 | 1 | 1 | 0.986 | 0.971 |
| complete_enclosure | 55 | 55 | 0 | 0 | 0 | 1.000 | 1.000 |

## Confusion matrix
See evaluation/v2_best_test/confusion_matrix.png (ultralytics output).

## Not assessed (no data / not done)
- Webcam/real-world performance: requires new recordings not used in training.
- Lighting/background/occlusion generalization: dataset is one video per class.
- Cap open/closed: no labels exist.