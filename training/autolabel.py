"""Bootstrap auto-labeling (PRELIMINARY - every proposal needs human review).

1. Trains a small YOLO on the frames a human has ALREADY labeled (annotations/).
2. Predicts boxes for the not-yet-verified frames and writes them to annotations_proposed/ (class x y w h conf).
Proposals are never read by validate/prepare/train. A frame becomes ground truth only when a human
saves it in the dashboard annotator (which writes to annotations/)."""
import sys, argparse, random, shutil
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-labeled", type=int, default=20, help="verified frames required per dataset")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--weights", help="skip bootstrap training and use these weights")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    a = ap.parse_args()

    verified = {n: sorted(p.stem for p in (ANNOT_DIR / n).glob("*.txt")) if (ANNOT_DIR / n).exists() else [] for n in DATASETS}
    print("verified frames:", {n: len(v) for n, v in verified.items()})
    short = {n: len(verified[n]) for n in a.datasets if len(verified[n]) < a.min_labeled}
    if short and not a.weights:
        sys.exit(f"Need >= {a.min_labeled} human-verified frames per dataset first. Short: {short}")

    from ultralytics import YOLO
    import torch
    dev = 0 if torch.cuda.is_available() else "cpu"

    if a.weights:
        weights = a.weights
    else:
        if BOOTSTRAP_DIR.exists():
            shutil.rmtree(BOOTSTRAP_DIR)
        rng = random.Random(0)
        for n in DATASETS:
            by_stem = {}
            for p in list_images(n):
                by_stem.setdefault(p.stem, p)
            stems = [s for s in verified[n] if s in by_stem]
            rng.shuffle(stems)
            nv = max(1, int(len(stems) * 0.15))
            for i, s in enumerate(stems):
                sp = "val" if i < nv else "train"
                (BOOTSTRAP_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
                (BOOTSTRAP_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)
                shutil.copy2(by_stem[s], BOOTSTRAP_DIR / "images" / sp / f"{n}__{s}{by_stem[s].suffix}")
                shutil.copy2(ANNOT_DIR / n / f"{s}.txt", BOOTSTRAP_DIR / "labels" / sp / f"{n}__{s}.txt")
        (BOOTSTRAP_DIR / "data.yaml").write_text(
            f"path: {BOOTSTRAP_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
            + "".join(f"  {k}: {v}\n" for k, v in CLASS_NAMES.items()))
        pre = ROOT / "models" / "pretrained" / "yolo11n.pt"
        m = YOLO(str(pre) if pre.exists() else "yolo11n.pt")
        m.train(data=str(BOOTSTRAP_DIR / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=dev,
                project=str(ROOT / "models" / "checkpoints"), name="bootstrap", exist_ok=True, patience=20,
                degrees=15, fliplr=0.5, mosaic=0.5, plots=False, seed=0)
        weights = str(ROOT / "models" / "checkpoints" / "bootstrap" / "weights" / "best.pt")

    model = YOLO(weights)
    total = 0
    for n in a.datasets:
        spec = DATASETS[n]
        done = set(verified[n])
        out = PROPOSED_DIR / n
        if out.exists():
            shutil.rmtree(out)  # stale proposals are replaced; verified labels are never touched
        out.mkdir(parents=True, exist_ok=True)
        seen, empty = set(), 0
        for p in list_images(n):
            if p.stem in done or p.stem in seen or p.stem in excluded(n) or p.stem in rejected(n):
                continue
            seen.add(p.stem)
            r = model.predict(str(p), conf=a.conf, imgsz=a.imgsz, device=dev, verbose=False)[0]
            best = {}
            for b in r.boxes:  # keep the single best box of each class this dataset may contain
                c = int(b.cls)
                if c in spec["allowed"] and (c not in best or float(b.conf) > best[c][0]):
                    best[c] = (float(b.conf), *b.xywhn[0].tolist())
            if not best:
                empty += 1
                continue
            (out / f"{p.stem}.txt").write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {cf:.4f}\n" for c, (cf, x, y, w, h) in best.items()))
            total += 1
        print(f"[{n}] proposals written: {len(seen) - empty}, no detection (label by hand): {empty}")
    print(f"Done: {total} PRELIMINARY proposals in {PROPOSED_DIR}. Review them in the dashboard annotator.")


if __name__ == "__main__":
    main()
