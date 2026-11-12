import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

MODELS_DIR = ROOT / "models" / "trained"
EVAL_DIR = ROOT / "evaluation"
LOG_DIR = ROOT / "logs"
CLASS_NAMES = {0: "body", 1: "top", 2: "complete_enclosure"}
COLORS_BGR = {"body": (255, 160, 40), "top": (60, 200, 255), "complete_enclosure": (80, 220, 80)}

# runtime-adjustable (POST /api/settings), no restart needed
SETTINGS = dict(conf=0.50, iou=0.45, imgsz=640, max_det=10, camera_index=0)
LIMITS = dict(conf=(0.10, 0.95), iou=(0.1, 0.9), imgsz=(160, 1280), max_det=(1, 50))
