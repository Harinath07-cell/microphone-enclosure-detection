"""Assemble evaluation/model_quality_report.md from REAL saved artifacts only (no numbers are invented).
Inputs: models/checkpoints/<run>/results.csv, evaluation/*_val.json, *_test.json, per_dataset_report.json."""
import sys, json, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="baseline")
a = ap.parse_args()
E = ROOT / "evaluation"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    import csv
    rows = []
    rp = ROOT / "models" / "checkpoints" / a.run / "results.csv"
    if rp.exists():
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(open(rp))]
    val, test, per = load(E / f"{a.run}_best_val.json"), load(E / f"{a.run}_best_test.json"), load(E / "per_dataset_report.json")
    L = [f"# Model quality report — run `{a.run}`", ""]
    L += ["## Data used", "```", dumps(load(UNIFIED_DIR / "split_meta.json")), "```",
          "All labels human-verified (annotations/); unlabeled frames were dropped by user choice. Splits are contiguous time blocks per video with 5-frame gaps.", ""]
    if rows:
        best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
        last = rows[-1]
        L += ["## Training curves (from results.csv)", f"- epochs run: {len(rows)}; best epoch (val mAP50-95): {int(best['epoch'])}",
              f"- train box/cls loss at last epoch: {last['train/box_loss']:.3f} / {last['train/cls_loss']:.3f}",
              f"- val box/cls loss at last epoch: {last['val/box_loss']:.3f} / {last['val/cls_loss']:.3f}",
              f"- val box/cls loss at best epoch: {best['val/box_loss']:.3f} / {best['val/cls_loss']:.3f}",
              f"- val min box loss {min(r['val/box_loss'] for r in rows):.3f} at epoch {int(min(rows, key=lambda r: r['val/box_loss'])['epoch'])}",
              f"- final val mAP50 {last['metrics/mAP50(B)']:.3f}, mAP50-95 {last['metrics/mAP50-95(B)']:.3f}", ""]
        gap = last["val/box_loss"] / max(last["train/box_loss"], 1e-9)
        up = last["val/box_loss"] > min(r["val/box_loss"] for r in rows) * 1.25
        L += ["## Over/underfitting analysis", f"- final val/train box-loss ratio: {gap:.2f}",
              f"- val box loss rose >25% above its minimum late in training: {'YES (overfitting sign)' if up else 'no'}",
              f"- train box loss still high (>1.5) and val mAP50-95 < 0.5 (underfitting sign): {'YES' if last['train/box_loss'] > 1.5 and last['metrics/mAP50-95(B)'] < 0.5 else 'no'}",
              "Note: validation frames come from the same videos as training, so a good val score does not prove generalization.", ""]
    for title, d in (("Validation (model-selection split)", val), ("Held-out test (final)", test)):
        L.append(f"## {title}")
        if not d:
            L += ["Not run.", ""]
            continue
        L += [f"- precision {d['precision']:.3f}, recall {d['recall']:.3f}, mAP50 {d['map50']:.3f}, mAP50-95 {d['map50_95']:.3f}",
              "", "| class | P | R | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
        L += [f"| {n} | {x['precision']:.3f} | {x['recall']:.3f} | {x['map50']:.3f} | {x['map50_95']:.3f} |" for n, x in d["per_class"].items()]
        L.append("")
    L.append("## Per-source-dataset test (conf 0.5, IoU 0.5)")
    if per:
        L += ["| dataset | images | TP | FP | FN | wrong class | precision | recall |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {n} | {r['images']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['wrong_class']} | {r['precision']:.3f} | {r['recall']:.3f} |" for n, r in per["datasets"].items()]
    else:
        L.append("Not run.")
    L += ["", "## Confusion matrix", f"See evaluation/{a.run}_best_test/confusion_matrix.png (ultralytics output).", "",
          "## Not assessed (no data / not done)", "- Webcam/real-world performance: requires new recordings not used in training.",
          "- Lighting/background/occlusion generalization: dataset is one video per class.", "- Cap open/closed: no labels exist."]
    (E / "model_quality_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
