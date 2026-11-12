"""Train YOLO11n on datasets/unified. Model selection uses validation metrics only; test set is not touched."""
import sys, argparse, shutil, datetime
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--degrees", type=float, default=15)
    ap.add_argument("--mosaic", type=float, default=0.5)
    a = ap.parse_args()
    if not (UNIFIED_DIR / "data.yaml").exists():
        sys.exit("datasets/unified missing. Run: python training/validate_dataset.py && python training/prepare_dataset.py")
    import torch
    from ultralytics import YOLO

    dev = 0 if torch.cuda.is_available() else "cpu"
    pre = ROOT / "models" / "pretrained" / a.model
    m = YOLO(str(pre) if pre.exists() else a.model)
    m.train(data=str(UNIFIED_DIR / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=dev,
            patience=a.patience, optimizer="AdamW", lr0=a.lr0, weight_decay=a.weight_decay, cos_lr=True,
            degrees=a.degrees, fliplr=a.fliplr, flipud=0.0, mosaic=a.mosaic, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
            scale=0.5, translate=0.1, close_mosaic=10, project=str(ROOT / "models" / "checkpoints"), name=a.name,
            exist_ok=True, plots=True, seed=0, deterministic=True, val=True)
    best = ROOT / "models" / "checkpoints" / a.name / "weights" / "best.pt"
    dst = ROOT / "models" / "trained" / f"{a.name}_best.pt"
    shutil.copy2(best, dst)
    (ROOT / "models" / "trained" / f"{a.name}_info.json").write_text(dumps(dict(
        args=vars(a), trained=datetime.datetime.now().isoformat(), device=str(dev),
        selected_by="validation fitness (0.1*mAP50 + 0.9*mAP50-95)")))
    print("saved", dst)


if __name__ == "__main__":
    main()
