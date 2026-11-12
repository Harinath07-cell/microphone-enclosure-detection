import time, threading, subprocess, sys, json, io
import cv2, numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from config import *
from camera import Camera
from detection import Detector, interpret
from metrics import LiveMetrics, saved_eval, read_json, training_curves
import common as C
from validate_dataset import validate

app = FastAPI(title="Microphone Enclosure Detector")
cam, det, live = Camera(), Detector(), LiveMetrics()
state = dict(detecting=False, dets=[], status="NO OBJECT DETECTED", jpeg=None, last_error=None)
train_proc = dict(p=None, started=None, log=None)
lock = threading.Lock()


def worker():
    last_id = -1
    while True:
        f = cam.frame
        if f is None or cam.frame_id == last_id:
            time.sleep(0.005)
            continue
        last_id = cam.frame_id
        dets, status = [], None
        try:
            if state["detecting"] and det.model is not None:
                dets, ms = det.predict(f)
                live.add(ms, time.time())
                status = interpret(dets)
                state["dets"], state["status"] = dets, status
        except Exception as e:  # keep the loop alive; surface the error
            state["last_error"] = str(e)
        out = Detector.draw(f, dets, status, live.snapshot()["detection_fps"] if status else None) if status else f
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            state["jpeg"] = buf.tobytes()


threading.Thread(target=worker, daemon=True).start()


@app.on_event("startup")
def _startup():
    det.load()  # model load only; training/camera never start automatically


# ---------------- status / controls ----------------
@app.get("/api/status")
def status():
    ev = saved_eval()
    return dict(
        camera=dict(connected=cam.running and cam.frame is not None, device=cam.index, resolution=cam.resolution,
                    capture_fps=round(cam.fps, 1), error=cam.error),
        detection=dict(running=state["detecting"], **live.snapshot(), status=state["status"] if state["detecting"] else "STOPPED",
                       objects=state["dets"] if state["detecting"] else [], count=len(state["dets"]) if state["detecting"] else 0,
                       error=state["last_error"]),
        model=dict(loaded=det.model is not None, weights=det.weights.name if det.weights else None, device=det.device, error=det.error),
        settings=SETTINGS, limits=LIMITS,
        cap_state=dict(state="NOT TRAINED", note="No open/closed labels exist in the datasets; capability not implemented."),
        evaluation=ev and dict(split=ev["split"], precision=ev["precision"], recall=ev["recall"], map50=ev["map50"], map50_95=ev["map50_95"]),
    )


@app.get("/api/cameras")
def cameras():
    was = cam.running
    if was:
        return dict(devices=[cam.index], note="camera in use; stop it to rescan")
    return dict(devices=Camera.probe())


class CamReq(BaseModel):
    index: Optional[int] = None


@app.post("/api/camera/start")
def camera_start(r: CamReq = CamReq()):
    if r.index is not None:
        SETTINGS["camera_index"] = r.index
    if cam.running and r.index is not None and r.index != cam.index:
        cam.stop()
    if not cam.start(SETTINGS["camera_index"]):
        raise HTTPException(500, cam.error)
    return dict(ok=True)


@app.post("/api/camera/stop")
def camera_stop():
    state["detecting"] = False
    cam.stop()
    state["jpeg"] = None
    return dict(ok=True)


@app.post("/api/detection/start")
def detection_start():
    if det.model is None and not det.load():
        raise HTTPException(409, det.error or "model not loaded")
    if not cam.running:
        raise HTTPException(409, "Start the camera first")
    state["detecting"] = True
    return dict(ok=True)


@app.post("/api/detection/stop")
def detection_stop():
    state["detecting"], state["dets"] = False, []
    return dict(ok=True)


@app.post("/api/model/reload")
def model_reload():
    return dict(ok=det.load(), error=det.error)


class Settings(BaseModel):
    conf: Optional[float] = None
    iou: Optional[float] = None
    imgsz: Optional[int] = None
    max_det: Optional[int] = None


@app.post("/api/settings")
def settings(s: Settings):
    for k, v in s.model_dump(exclude_none=True).items():
        lo, hi = LIMITS[k]
        SETTINGS[k] = type(SETTINGS[k])(min(max(v, lo), hi))
    return SETTINGS


@app.get("/stream")
def stream():
    def gen():
        last = None
        while True:
            j = state["jpeg"]
            if j is not None and j is not last:
                last = j
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            else:
                time.sleep(0.01)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


# ---------------- image test ----------------
@app.post("/api/test-image")
async def test_image(file: UploadFile = File(...)):
    if det.model is None and not det.load():
        raise HTTPException(409, det.error)
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "unreadable image")
    dets, ms = det.predict(img)
    ok, buf = cv2.imencode(".jpg", Detector.draw(img, dets))
    import base64
    return dict(detections=dets, status=interpret(dets), latency_ms=round(ms, 1),
                annotated="data:image/jpeg;base64," + base64.b64encode(buf).decode())


# ---------------- datasets / annotation ----------------
@app.get("/api/datasets")
def datasets():
    p = EVAL_DIR / "dataset_validation.json"
    return json.loads(p.read_text()) if p.exists() else dict(all_ready=False, datasets={}, note="not validated yet")


@app.post("/api/datasets/validate")
def datasets_validate():
    ok, rep = validate()
    return dict(all_ready=ok, datasets=rep)


def _img(ds, stem):
    for p in C.list_images(ds):
        if p.stem == stem:
            return p
    raise HTTPException(404)


@app.get("/api/annotate/{ds}/images")
def ann_list(ds: str):
    if ds not in C.DATASETS:
        raise HTTPException(404)
    stems = sorted({p.stem for p in C.list_images(ds)}, key=lambda s: int(s.split("_")[-1]))
    done = {p.stem for p in (C.ANNOT_DIR / ds).glob("*.txt")} if (C.ANNOT_DIR / ds).exists() else set()
    prop = {}
    if (C.PROPOSED_DIR / ds).exists():
        for p in (C.PROPOSED_DIR / ds).glob("*.txt"):
            cs = [float(l.split()[5]) for l in p.read_text().splitlines() if len(l.split()) == 6]
            prop[p.stem] = min(cs) if cs else 0.0
    exc, rej = C.excluded(ds), C.rejected(ds)
    return dict(allowed=sorted(C.DATASETS[ds]["allowed"]), classes=C.CLASS_NAMES,
                images=[dict(stem=s, labeled=s in done, excluded=s in exc, rejected=s in rej and s not in done,
                             proposed=s in prop and s not in done and s not in rej, conf=prop.get(s)) for s in stems])


@app.post("/api/annotate/{ds}/reject/{stem}")
def ann_reject(ds: str, stem: str):
    """Discard a model proposal. The frame stays unlabeled (label it by hand or exclude it)."""
    _img(ds, stem)
    s = C.rejected(ds) | {stem}
    C.write_set(C.PROPOSED_DIR / f"_rejected_{ds}.txt", s)
    pp = C.PROPOSED_DIR / ds / (stem + ".txt")
    if pp.exists():
        pp.unlink()
    return dict(ok=True)


@app.post("/api/annotate/{ds}/exclude/{stem}")
def ann_exclude(ds: str, stem: str, undo: bool = False):
    """Mark a frame unusable (blurred, object cut off...). Excluded frames never enter training."""
    _img(ds, stem)
    s = C.excluded(ds)
    s = s - {stem} if undo else s | {stem}
    C.write_set(C.ANNOT_DIR / f"_excluded_{ds}.txt", s)
    return dict(ok=True)


@app.get("/api/annotate/{ds}/image/{stem}")
def ann_image(ds: str, stem: str):
    return FileResponse(_img(ds, stem))


@app.get("/api/annotate/{ds}/label/{stem}")
def ann_get(ds: str, stem: str):
    lp = C.ANNOT_DIR / ds / (stem + ".txt")
    if lp.exists():
        return dict(proposed=False, boxes=[dict(cls=c, x=x, y=y, w=w, h=h) for c, x, y, w, h in C.read_label(lp)[0]])
    pp = C.PROPOSED_DIR / ds / (stem + ".txt")
    boxes = []
    if pp.exists():  # UNVERIFIED model proposal; becomes ground truth only when the user saves it
        for l in pp.read_text().splitlines():
            v = l.split()
            if len(v) == 6:
                boxes.append(dict(cls=int(v[0]), x=float(v[1]), y=float(v[2]), w=float(v[3]), h=float(v[4]), conf=float(v[5])))
    return dict(proposed=bool(boxes), boxes=boxes)


_auto = dict(p=None)


@app.post("/api/annotate/autolabel")
def autolabel_start():
    if _auto["p"] and _auto["p"].poll() is None:
        raise HTTPException(409, "auto-label already running")
    LOG_DIR.mkdir(exist_ok=True)
    _auto["p"] = subprocess.Popen([sys.executable, str(ROOT / "training/autolabel.py")], cwd=ROOT,
                                  stdout=open(LOG_DIR / "autolabel.log", "w"), stderr=subprocess.STDOUT)
    return dict(ok=True)


@app.get("/api/annotate/autolabel/status")
def autolabel_status():
    p = _auto["p"]
    lg = LOG_DIR / "autolabel.log"
    tail = lg.read_text(errors="ignore").splitlines()[-6:] if lg.exists() else []
    return dict(running=bool(p and p.poll() is None), exit_code=None if p is None or p.poll() is None else p.returncode, tail=tail)


class Boxes(BaseModel):
    boxes: list[dict]


@app.put("/api/annotate/{ds}/label/{stem}")
def ann_put(ds: str, stem: str, b: Boxes):
    _img(ds, stem)
    allowed = C.DATASETS[ds]["allowed"]
    lines = []
    for o in b.boxes:
        c = int(o["cls"])
        if c not in allowed:
            raise HTTPException(400, f"class {c} not allowed for {ds}")
        x, y, w, h = (float(o[k]) for k in "xywh")
        if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            raise HTTPException(400, "bad box")
        lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    if not lines:
        raise HTTPException(400, "empty annotation not allowed (delete instead)")
    d = C.ANNOT_DIR / ds
    d.mkdir(parents=True, exist_ok=True)
    (d / (stem + ".txt")).write_text("\n".join(lines) + "\n")
    return dict(ok=True)


# ---------------- training ----------------
@app.post("/api/training/start")
def train_start(epochs: int = 100, name: str = "baseline"):
    p = train_proc["p"]
    if p and p.poll() is None:
        raise HTTPException(409, "already training")
    if not (C.UNIFIED_DIR / "data.yaml").exists():
        raise HTTPException(409, "datasets/unified missing - labels must be completed, then run prepare_dataset.py")
    LOG_DIR.mkdir(exist_ok=True)
    log = open(LOG_DIR / f"train_{name}.log", "w")
    train_proc.update(p=subprocess.Popen([sys.executable, str(ROOT / "training/train.py"), "--epochs", str(epochs), "--name", name],
                                         stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), started=time.time(), log=name)
    return dict(ok=True)


@app.post("/api/training/stop")
def train_stop():
    p = train_proc["p"]
    if p and p.poll() is None:
        p.terminate()
    return dict(ok=True)


@app.get("/api/training/status")
def train_status():
    p = train_proc["p"]
    running = bool(p and p.poll() is None)
    cur = training_curves(train_proc["log"] or "baseline")
    return dict(running=running, exit_code=None if running or p is None else p.returncode,
                duration_s=round(time.time() - train_proc["started"]) if train_proc["started"] else 0,
                epochs_done=len(cur) if cur else 0, curves=cur,
                best_map50_95=max((r.get("metrics/mAP50-95(B)", 0) for r in cur), default=None) if cur else None)


@app.post("/api/evaluation/run")
def eval_run(split: str = "val"):
    w = det.find_weights()
    if not w:
        raise HTTPException(409, "no trained weights")
    r = subprocess.run([sys.executable, str(ROOT / "training/evaluate.py"), "--weights", str(w), "--split", split],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise HTTPException(500, r.stderr[-800:])
    return saved_eval()


@app.get("/api/evaluation")
def evaluation():
    return dict(latest=saved_eval(), per_dataset=read_json("per_dataset_report.json"), curves=training_curves())


app.mount("/", StaticFiles(directory=ROOT / "frontend" / "dashboard", html=True), name="ui")
