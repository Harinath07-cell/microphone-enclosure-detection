"""Single-owner webcam wrapper: one capture thread, one VideoCapture, released on stop."""
import threading, time, cv2


class Camera:
    def __init__(self):
        self.cap = None
        self.index = 0
        self.running = False
        self.frame = None
        self.frame_id = 0
        self.fps = 0.0
        self.resolution = None
        self.error = None
        self._lock = threading.Lock()
        self._thread = None

    def _open(self):
        # CAP_DSHOW opens quickly on Windows; fall back to default backend
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def start(self, index=None):
        with self._lock:
            if self.running:
                return True
            if index is not None:
                self.index = index
            cap = self._open()
            if cap is None:
                self.error = f"Cannot open camera {self.index}"
                return False
            self.cap, self.error, self.running = cap, None, True
            self.resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True

    def _loop(self):
        t0, n, fails = time.time(), 0, 0
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails > 30:  # lost device -> try to reconnect
                    self.error = "Camera disconnected - reconnecting"
                    self.cap.release()
                    time.sleep(1.0)
                    cap = self._open()
                    if cap is not None:
                        self.cap, self.error, fails = cap, None, 0
                time.sleep(0.02)
                continue
            fails = 0
            self.frame, self.frame_id = f, self.frame_id + 1
            n += 1
            if time.time() - t0 >= 1.0:
                self.fps, n, t0 = n / (time.time() - t0), 0, time.time()

    def stop(self):
        self.running = False
        t = self._thread
        if t:
            t.join(timeout=3)
        with self._lock:
            if self.cap is not None:
                self.cap.release()
            self.cap, self.frame, self.fps, self.resolution = None, None, 0.0, None

    @staticmethod
    def probe(max_idx=4):
        found = []
        for i in range(max_idx):
            c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if c.isOpened():
                found.append(i)
            c.release()
        return found
