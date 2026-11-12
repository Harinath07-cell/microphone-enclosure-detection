"""Evaluate a model on a split of the unified dataset -> evaluation/*.json (+ ultralytics plots).
Use --split val for model selection; use --split test only for the FINAL report."""
import sys, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/trained/baseline_best.pt"))
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    a = ap.parse_args()
    from ultralytics import YOLO

    m = YOLO(a.weights)
    stem = Path(a.weights).stem
    r = m.val(data=str(UNIFIED_DIR / "data.yaml"), split=a.split, imgsz=a.imgsz, conf=a.conf, plots=True,
              project=str(ROOT / "evaluation"), name=f"{stem}_{a.split}", exist_ok=True)
    out = dict(weights=a.weights, split=a.split, precision=float(r.box.mp), recall=float(r.box.mr),
               map50=float(r.box.map50), map50_95=float(r.box.map), speed_ms=r.speed, per_class={})
    for i, c in enumerate(r.box.ap_class_index):
        p, rc, ap50, ap_ = r.box.class_result(i)
        out["per_class"][CLASS_NAMES[int(c)]] = dict(precision=float(p), recall=float(rc), map50=float(ap50), map50_95=float(ap_))
    (ROOT / "evaluation" / "latest_eval.json").write_text(dumps(out))
    (ROOT / "evaluation" / f"{stem}_{a.split}.json").write_text(dumps(out))
    print(dumps(out))


if __name__ == "__main__":
    main()
