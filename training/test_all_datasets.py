"""Per-source-dataset evaluation on that dataset's held-out block, with TP/FP/FN/wrong-class counts.
Writes evaluation/per_dataset_report.{json,md}. Tunes nothing."""
import sys, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(ROOT / "models/trained/baseline_best.pt"))
ap.add_argument("--split", default="test")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--iou", type=float, default=0.5)
a = ap.parse_args()
from ultralytics import YOLO


def iou(u, v):
    ax1, ay1, ax2, ay2 = u[1] - u[3] / 2, u[2] - u[4] / 2, u[1] + u[3] / 2, u[2] + u[4] / 2
    bx1, by1, bx2, by2 = v[1] - v[3] / 2, v[2] - v[4] / 2, v[1] + v[3] / 2, v[2] + v[4] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    i = iw * ih
    return i / (u[3] * u[4] + v[3] * v[4] - i + 1e-9)


m = YOLO(a.weights)
res = {}
for name in DATASETS:
    imgs = sorted((UNIFIED_DIR / "images" / a.split).glob(f"{name}__*"))
    tp = fp = fn = wrong = 0
    fails = []
    for p in imgs:
        gt = read_label(UNIFIED_DIR / "labels" / a.split / (p.stem + ".txt"))[0]
        pr = m.predict(str(p), conf=a.conf, verbose=False)[0]
        pd = [(int(b.cls), *b.xywhn[0].tolist()) for b in pr.boxes]
        used, bad = set(), False
        for g in gt:
            best = max(((iou(g, q), j) for j, q in enumerate(pd) if j not in used), default=(0, -1))
            if best[0] >= a.iou:
                used.add(best[1])
                if pd[best[1]][0] == g[0]:
                    tp += 1
                else:
                    wrong += 1; bad = True
            else:
                fn += 1; bad = True
        extra = len(pd) - len(used)
        fp += extra
        bad |= extra > 0
        if bad:
            fails.append(p.name)
    res[name] = dict(images=len(imgs), tp=tp, fp=fp, fn=fn, wrong_class=wrong,
                     precision=tp / max(tp + fp + wrong, 1), recall=tp / max(tp + fn + wrong, 1), failing_images=fails[:50])
(ROOT / "evaluation" / "per_dataset_report.json").write_text(dumps(dict(weights=a.weights, split=a.split, conf=a.conf, datasets=res)))
md = [f"# Per-dataset report ({a.split} split, conf={a.conf}, IoU={a.iou})\n",
      "| dataset | images | TP | FP | FN | wrong class | precision | recall |", "|---|---|---|---|---|---|---|---|"]
md += [f"| {n} | {r['images']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['wrong_class']} | {r['precision']:.3f} | {r['recall']:.3f} |" for n, r in res.items()]
(ROOT / "evaluation" / "per_dataset_report.md").write_text("\n".join(md))
print("\n".join(md))
