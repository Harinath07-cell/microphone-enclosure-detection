"""Validate all source datasets + human annotations. Never fabricates labels.
Writes evaluation/dataset_validation.json. Exit code 1 if anything blocks training."""
import sys, cv2, collections
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *


def validate():
    report, ok = {}, True
    for name, spec in DATASETS.items():
        d = spec["dir"]
        r = dict(exists=d.exists(), images=0, unreadable=[], resolutions={}, split_counts={},
                 duplicates_within=0, cross_split_duplicate_images=0, labeled=0, missing_labels=0,
                 empty_labels=0, invalid_labels={}, wrong_class={}, missing_required_class=0, class_counts={})
        if not d.exists():
            report[name] = r; ok = False; continue
        imgs = list_images(name)
        r["images"] = len(imgs)
        res = collections.Counter(); sp = collections.Counter()
        seen = {}; cross = 0; dup = 0
        for p in imgs:
            sp[p.parent.name] += 1
            im = cv2.imread(str(p))
            if im is None:
                r["unreadable"].append(p.name); continue
            res[f"{im.shape[1]}x{im.shape[0]}"] += 1
            h = md5(p)
            if h in seen:
                dup += 1
                if seen[h] != p.parent.name: cross += 1
            else:
                seen[h] = p.parent.name
        # same frame number appearing in several splits (frames duplicated under different names/splits)
        by_stem = collections.defaultdict(set)
        for p in imgs: by_stem[p.stem].add(p.parent.name)
        r["stems_in_multiple_splits"] = sum(len(v) > 1 for v in by_stem.values())
        r["resolutions"] = dict(res); r["split_counts"] = dict(sp)
        r["duplicates_within"] = dup; r["cross_split_duplicate_images"] = cross
        cc = collections.Counter()
        exc = excluded(name)
        r["excluded"] = len(exc & set(by_stem))
        for stem in sorted(set(by_stem) - exc):
            lp = ANNOT_DIR / name / (stem + ".txt")
            if not lp.exists():
                r["missing_labels"] += 1; continue
            rows, errs = read_label(lp)
            if errs: r["invalid_labels"][stem] = errs
            if not rows: r["empty_labels"] += 1; continue
            r["labeled"] += 1
            cls = {c for c, *_ in rows}
            bad = cls - spec["allowed"]
            if bad: r["wrong_class"][stem] = sorted(bad)
            if not spec["required"] <= cls: r["missing_required_class"] += 1
            for c, *_ in rows: cc[CLASS_NAMES.get(c, str(c))] += 1
        r["class_counts"] = dict(cc)
        r["unique_frames"] = len(by_stem) - r["excluded"]
        r["ready_for_training"] = bool(r["missing_labels"] == 0 and r["empty_labels"] == 0 and not r["invalid_labels"]
                                       and not r["wrong_class"] and not r["missing_required_class"] and not r["unreadable"])
        ok &= r["ready_for_training"]
        report[name] = r
    (ROOT / "evaluation").mkdir(exist_ok=True)
    (ROOT / "evaluation" / "dataset_validation.json").write_text(dumps(dict(all_ready=ok, datasets=report)))
    return ok, report


if __name__ == "__main__":
    ok, rep = validate()
    for n, r in rep.items():
        print(f"[{n}] images={r['images']} unique_frames={r.get('unique_frames')} splits={r['split_counts']} "
              f"cross-split dup imgs={r['cross_split_duplicate_images']} stems in >1 split={r.get('stems_in_multiple_splits')}\n"
              f"    labeled={r['labeled']} missing_labels={r['missing_labels']} empty={r['empty_labels']} "
              f"invalid={len(r['invalid_labels'])} wrong_class={len(r['wrong_class'])} ready={r.get('ready_for_training')}")
    print("\nALL DATASETS READY" if ok else "\nNOT READY: human-verified labels are missing/invalid (see evaluation/dataset_validation.json)")
    sys.exit(0 if ok else 1)
