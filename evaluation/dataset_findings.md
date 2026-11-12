# Dataset findings (from `training/validate_dataset.py`, machine output in dataset_validation.json)

| Dataset | Images | Unique frames | Splits (train/val/test) | Cross-split duplicate images | Labels |
|---|---|---|---|---|---|
| body | 492 | 444 | 313/110/69 | 48 | **0** |
| top | 522 | 450 | 315/126/81 | 72 | **0** |
| top_body_separate (`mic_enclosure_dataset/`) | 406 | 406 | 293/78/35 | 0 | **0** |
| complete_enclosure | 454 | 406 | 287/110/57 | 48 | **0** |

All images readable; all 832×464. Zero YOLO label files exist anywhere, so supervised training is blocked.
Counts differ from the earlier report (README says 444/450/…; folders hold more because val/test share duplicated frames).

Blockers: human bounding-box annotation (~1,700 frames); sessions beyond one video per class for real generalization testing;
cap open/closed data. Model-quality acceptance criteria cannot be evaluated until a model is trained on verified labels.
