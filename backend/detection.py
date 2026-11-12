import time, cv2
from config import *


def interpret(dets):
    names = {d["class"] for d in dets}
    if "complete_enclosure" in names:
        return "COMPLETE ENCLOSURE"
    if "top" in names and "body" in names:
        return "TOP + BODY SEPARATE"
    if "top" in names:
        return "TOP DETECTED"
    if "body" in names:
        return "BODY DETECTED"
    return "NO OBJECT DETECTED"


class Detector:
    def __init__(self):
        self.model = None
        self.weights = None
        self.device = "cpu"
        self.error = None

    def find_weights(self):
        c = sorted(MODELS_DIR.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return c[0] if c else None

    def load(self):
        w = self.find_weights()
        if w is None:
            self.error = "No trained weights in models/trained (train a model first)"
            return False
        try:
            import torch
            from ultralytics import YOLO
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(str(w))
            self.model.to(self.device)
            self.weights, self.error = w, None
            return True
        except Exception as e:
            self.model, self.error = None, str(e)
            return False

    def predict(self, frame):
        """Returns (detections, latency_ms)."""
        s = SETTINGS
        t = time.perf_counter()
        r = self.model.predict(frame, conf=s["conf"], iou=s["iou"], imgsz=s["imgsz"], max_det=s["max_det"],
                               device=self.device, verbose=False)[0]
        ms = (time.perf_counter() - t) * 1000
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append(dict(**{"class": CLASS_NAMES[int(b.cls)]}, confidence=float(b.conf),
                            x=round(x1), y=round(y1), width=round(x2 - x1), height=round(y2 - y1)))
        return out, ms

    @staticmethod
    def draw(frame, dets, status=None, fps=None):
        f = frame.copy()
        for d in dets:
            c = COLORS_BGR.get(d["class"], (255, 255, 255))
            p1, p2 = (d["x"], d["y"]), (d["x"] + d["width"], d["y"] + d["height"])
            cv2.rectangle(f, p1, p2, c, 2)
            label = f'{d["class"]} {d["confidence"] * 100:.1f}%'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(f, (p1[0], p1[1] - th - 8), (p1[0] + tw + 6, p1[1]), c, -1)
            cv2.putText(f, label, (p1[0] + 3, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if status:
            cv2.putText(f, status + (f"  {fps:.1f} FPS" if fps else ""), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 230, 80), 2)
        return f
