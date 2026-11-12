"""Launch the dashboard: python run.py [--port 8000]"""
import sys, argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "backend"))
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="127.0.0.1")
a = ap.parse_args()
import uvicorn
uvicorn.run("main:app", host=a.host, port=a.port, app_dir=str(ROOT / "backend"))
