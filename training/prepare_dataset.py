"""Build datasets/unified from the human-verified annotations.
- refuses to run unless validation passes (no fabricated/missing labels)
- de-duplicates identical frames, then splits each source video into CONTIGUOUS time blocks
  (train | gap | val | gap | test) so near-identical adjacent frames never straddle splits.
- original datasets are untouched (images are copied)."""
import sys, shutil, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
from validate_dataset import validate

ap = argparse.ArgumentParser()
ap.add_argument("--val", type=float, default=0.15)
ap.add_argument("--test", type=float, default=0.15)
ap.add_argument("--gap", type=int, default=5, help="frames dropped between blocks")
a = ap.parse_args()

ok, _ = validate()
if not ok:
    sys.exit("Refusing to build: validation failed. Annotate in the dashboard (Dataset Management > Annotate) first.")
if UNIFIED_DIR.exists():
    shutil.rmtree(UNIFIED_DIR)
meta = {}
for name in DATASETS:
    seen, items = set(), []
    exc = excluded(name)
    for p in sorted(list_images(name), key=frame_no):
        if p.stem in exc:
            continue
        h = md5(p)
        if h in seen:
            continue
        seen.add(h)
        items.append(p)
    n, g = len(items), a.gap
    nt, nv = int(n * a.test), int(n * a.val)
    ntr = n - nt - nv - 2 * g
    blocks = {"train": items[:ntr], "val": items[ntr + g: ntr + g + nv], "test": items[ntr + 2 * g + nv:]}
    for sp, lst in blocks.items():
        (UNIFIED_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
        (UNIFIED_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for p in lst:
            stem = f"{name}__{p.stem}"
            shutil.copy2(p, UNIFIED_DIR / "images" / sp / (stem + p.suffix))
            shutil.copy2(label_path(name, p), UNIFIED_DIR / "labels" / sp / (stem + ".txt"))
    meta[name] = {k: len(v) for k, v in blocks.items()}
(UNIFIED_DIR / "data.yaml").write_text(
    f"path: {UNIFIED_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    + "".join(f"  {k}: {v}\n" for k, v in CLASS_NAMES.items()))
(UNIFIED_DIR / "split_meta.json").write_text(dumps(meta))
print(dumps(meta))
