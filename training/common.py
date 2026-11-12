"""Shared constants/helpers. Unified class IDs: 0 body, 1 top, 2 complete_enclosure."""
import hashlib, re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = {0: "body", 1: "top", 2: "complete_enclosure"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# name -> (source folder, allowed unified class ids, required class ids per image)
DATASETS = {
    "body": dict(dir=ROOT / "body_dataset", allowed={0}, required={0}),
    "top": dict(dir=ROOT / "top_dataset", allowed={1}, required={1}),
    "top_body_separate": dict(dir=ROOT / "mic_enclosure_dataset" / "mic_enclosure_dataset", allowed={0, 1}, required=set()),  # user-confirmed: frames may show only one of the two parts
    "complete_enclosure": dict(dir=ROOT / "complete_enclosure_dataset", allowed={2}, required={2}),
}
# Human-verified YOLO labels (unified class IDs) live here: annotations/<dataset>/<image_stem>.txt
import os
ANNOT_DIR = Path(os.environ.get("MIC_ANNOT_DIR", ROOT / "annotations"))
# Model-proposed boxes (UNVERIFIED, 6th column = confidence). Never used for training/validation until a human saves them into ANNOT_DIR.
PROPOSED_DIR = Path(os.environ.get("MIC_PROPOSED_DIR", ROOT / "annotations_proposed"))
BOOTSTRAP_DIR = Path(os.environ.get("MIC_BOOTSTRAP_DIR", ROOT / "datasets" / "bootstrap"))
UNIFIED_DIR = ROOT / "datasets" / "unified"


def list_images(name):
    d = DATASETS[name]["dir"] / "images"
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def frame_no(path):
    m = re.search(r"(\d+)$", Path(path).stem)
    return int(m.group(1)) if m else 0


NL = chr(10)


def _read_set(p):
    return set(Path(p).read_text().split()) if Path(p).exists() else set()


def excluded(name):
    """Frames a human marked unusable (blurred/occluded/no object). Excluded from training entirely."""
    # _unlabeled_*: frames dropped because the user chose to train only on frames they verified (delete the file to undo)
    return _read_set(ANNOT_DIR / f"_excluded_{name}.txt") | _read_set(ANNOT_DIR / f"_unlabeled_{name}.txt")


def rejected(name):
    """Model proposals a human rejected; not re-proposed."""
    return _read_set(PROPOSED_DIR / f"_rejected_{name}.txt")


def write_set(p, s):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(NL.join(sorted(s)) + (NL if s else ""))


def label_path(name, img):
    return ANNOT_DIR / name / (Path(img).stem + ".txt")


def read_label(path):
    """Returns (rows, errors). rows = [(cls,x,y,w,h)]"""
    rows, errs = [], []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        p = line.split()
        try:
            if len(p) != 5:
                raise ValueError("expected 5 fields")
            c = int(p[0]); x, y, w, h = map(float, p[1:])
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("coords out of range")
            if x - w / 2 < -1e-3 or x + w / 2 > 1 + 1e-3 or y - h / 2 < -1e-3 or y + h / 2 > 1 + 1e-3:
                raise ValueError("box exceeds image")
            rows.append((c, x, y, w, h))
        except Exception as e:
            errs.append(f"line {i}: {e}")
    return rows, errs


def dumps(o):
    return json.dumps(o, indent=2, default=str)
