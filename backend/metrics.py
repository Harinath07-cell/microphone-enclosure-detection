"""Live perf metrics + saved-evaluation lookup. Live precision/recall are NOT computed (no live ground truth)."""
import json, collections, csv
from config import *


class LiveMetrics:
    def __init__(self):
        self.lat = collections.deque(maxlen=60)
        self.ts = collections.deque(maxlen=60)

    def add(self, ms, t):
        self.lat.append(ms)
        self.ts.append(t)

    def snapshot(self):
        lat = list(self.lat)
        fps = (len(self.ts) - 1) / (self.ts[-1] - self.ts[0]) if len(self.ts) > 1 and self.ts[-1] > self.ts[0] else 0.0
        return dict(latency_ms=round(sum(lat) / len(lat), 1) if lat else None, detection_fps=round(fps, 1))


def saved_eval():
    p = EVAL_DIR / "latest_eval.json"
    return json.loads(p.read_text()) if p.exists() else None


def read_json(name):
    p = EVAL_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def training_curves(run="baseline"):
    p = ROOT / "models" / "checkpoints" / run / "results.csv"
    if not p.exists():
        return None
    with open(p) as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    keys = ["epoch", "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss",
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    return [{k: float(r[k]) for k in keys if k in r} for r in rows]
