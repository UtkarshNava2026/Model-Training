#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Pre-training sanity check for the VOC dataset used by exps/example/yolox_voc/*.

Checks every XML in train/valid splits for things that silently hurt training or
crash YOLOX mid-run, and prints per-class stats with extra detail for `disinfection`.

Usage:
    python tools/check_voc_dataset.py                       # uses $YOLOX_DATADIR or ./datasets
    python tools/check_voc_dataset.py --voc-root /path/VOCdevkit/VOC2012
    python tools/check_voc_dataset.py --check-images        # also opens every image (slower)

Exit code 1 if any blocking error is found.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_voc_classes():
    # load the file directly so this script needs no torch / yolox install
    import importlib.util
    p = os.path.join(ROOT, "yolox", "data", "datasets", "voc_classes.py")
    spec = importlib.util.spec_from_file_location("voc_classes", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VOC_CLASSES


VOC_CLASSES = _load_voc_classes()

FOCUS = "disinfection"


def pct(vals, q):
    if not vals:
        return 0
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--check-images", action="store_true")
    ap.add_argument("--input-size", type=int, default=960,
                    help="model input (square). Box sizes are reported after letterbox resize to this")
    args = ap.parse_args()

    if args.voc_root is None:
        base = os.getenv("YOLOX_DATADIR", os.path.join(ROOT, "datasets"))
        args.voc_root = os.path.join(base, "VOCdevkit", "VOC2012")
    voc = args.voc_root
    devkit = os.path.dirname(voc)
    print(f"VOC root : {voc}")
    print(f"Classes  : {len(VOC_CLASSES)} -> {list(VOC_CLASSES)}\n")

    errors, warns = [], []
    split_ids = {}
    for sp in args.splits:
        f = os.path.join(voc, "ImageSets", "Main", sp + ".txt")
        if not os.path.isfile(f):
            errors.append(f"missing split file {f}")
            continue
        ids = [ln.strip() for ln in open(f) if ln.strip()]
        split_ids[sp] = ids
        dup = [k for k, v in Counter(ids).items() if v > 1]
        if dup:
            warns.append(f"{sp}: {len(dup)} duplicated ids (ok only if oversampling on purpose)")

    if "train" in split_ids and "valid" in split_ids:
        leak = set(split_ids["train"]) & set(split_ids["valid"])
        if leak:
            errors.append(f"{len(leak)} ids are in BOTH train and valid, e.g. {sorted(leak)[:3]}")

    for sp, ids in split_ids.items():
        boxes = Counter()
        imgs = Counter()
        negatives = 0
        unknown = Counter()
        float_coords = 0
        missing_tags = 0
        bad_boxes = []
        tiny = Counter()          # min side < 8 px at model input
        minside = defaultdict(list)  # min side at model input, per class
        dis_w, dis_h, dis_area_frac, dis_per_img = [], [], [], []
        ar_dis = []

        for img_id in dict.fromkeys(ids):
            xp = os.path.join(voc, "Annotations", img_id + ".xml")
            if not os.path.isfile(xp):
                errors.append(f"{sp}: missing XML {xp}")
                continue
            try:
                root = ET.parse(xp).getroot()
            except ET.ParseError as e:
                errors.append(f"{sp}: cannot parse {xp}: {e}")
                continue

            size = root.find("size")
            W = int(float(size.findtext("width", "0"))) if size is not None else 0
            H = int(float(size.findtext("height", "0"))) if size is not None else 0
            if W <= 0 or H <= 0:
                warns.append(f"{sp}: {img_id} has width/height 0 in XML")

            if args.check_images:
                import cv2
                jp = os.path.join(voc, "JPEGImages", img_id + ".jpg")
                if not os.path.isfile(jp):
                    jp = os.path.splitext(jp)[0] + ".jpeg"
                im = cv2.imread(jp)
                if im is None:
                    errors.append(f"{sp}: image missing/unreadable for {img_id}")
                elif W > 0 and (im.shape[1] != W or im.shape[0] != H):
                    errors.append(
                        f"{sp}: {img_id} XML size {W}x{H} != image {im.shape[1]}x{im.shape[0]}"
                    )

            objs = root.findall("object")
            if not objs:
                negatives += 1
                continue
            seen = set()
            n_dis = 0
            for o in objs:
                name = (o.findtext("name") or "").strip()
                if name not in VOC_CLASSES:
                    unknown[name] += 1
                    continue
                for t in ("difficult", "truncated", "pose"):
                    if o.find(t) is None:
                        missing_tags += 1
                        break
                bb = o.find("bndbox")
                raw = [bb.findtext(k, "") for k in ("xmin", "ymin", "xmax", "ymax")]
                if any("." in r for r in raw):
                    float_coords += 1
                try:
                    x1, y1, x2, y2 = [float(r) for r in raw]
                except ValueError:
                    errors.append(f"{sp}: {img_id} non-numeric bbox {raw}")
                    continue
                if x2 <= x1 or y2 <= y1:
                    bad_boxes.append((img_id, name, raw))
                    continue
                if W > 0 and (x1 < 0 or y1 < 0 or x2 > W + 1 or y2 > H + 1):
                    warns.append(f"{sp}: {img_id} {name} box outside image {raw} vs {W}x{H}")
                w, h = x2 - x1, y2 - y1
                r = args.input_size / max(W, H) if W > 0 else 1.0
                ms = min(w, h) * r
                minside[name].append(ms)
                if ms < 8:
                    tiny[name] += 1
                boxes[name] += 1
                seen.add(name)
                if name == FOCUS:
                    n_dis += 1
                    dis_w.append(w)
                    dis_h.append(h)
                    ar_dis.append(w / h)
                    if W > 0:
                        dis_area_frac.append(w * h / (W * H))
            for n in seen:
                imgs[n] += 1
            if n_dis:
                dis_per_img.append(n_dis)

        n_unique = len(dict.fromkeys(ids))
        print(f"===== split: {sp}  ({len(ids)} ids, {n_unique} unique) =====")
        print(f"box min-side measured at model input {args.input_size}x{args.input_size}")
        print(f"{'class':<16}{'images':>8}{'boxes':>8}{'p10px':>7}{'p50px':>7}{'<8px':>7}{'<8%':>6}")
        for c in VOC_CLASSES:
            flag = "   <-- NO SAMPLES" if boxes[c] == 0 else ""
            frac = tiny[c] / max(boxes[c], 1)
            if boxes[c] and frac > 0.2:
                flag = "   <-- many too small to learn"
                warns.append(f"{sp}: {frac:.0%} of '{c}' boxes have min side <8 px at "
                             f"{args.input_size} input (stride-8 head limit)")
            print(f"{c:<16}{imgs[c]:>8}{boxes[c]:>8}{pct(minside[c], .1):>7.0f}"
                  f"{pct(minside[c], .5):>7.0f}{tiny[c]:>7}{frac:>6.0%}{flag}")
            if boxes[c] == 0:
                (errors if sp == "train" else warns).append(f"{sp}: class '{c}' has 0 boxes")
        print(f"{'negative imgs':<16}{negatives:>8}   ({negatives / max(n_unique, 1):.1%})")
        if unknown:
            errors.append(f"{sp}: labels not in VOC_CLASSES (training will KeyError): {dict(unknown)}")
        if bad_boxes:
            errors.append(f"{sp}: {len(bad_boxes)} zero/negative-size boxes, e.g. {bad_boxes[:3]}")
        if float_coords:
            print(f"note: {float_coords} boxes have float coords (handled by patched voc_eval)")
        if missing_tags:
            print(f"note: {missing_tags} objects miss difficult/truncated/pose (handled by patched voc_eval)")

        if dis_w:
            print(f"\n  [{FOCUS}] boxes={len(dis_w)} in {len(dis_per_img)} imgs, "
                  f"per-img avg={sum(dis_per_img) / len(dis_per_img):.2f}")
            print(f"  width  px p10/p50/p90 : {pct(dis_w, .1):.0f} / {pct(dis_w, .5):.0f} / {pct(dis_w, .9):.0f}")
            print(f"  height px p10/p50/p90 : {pct(dis_h, .1):.0f} / {pct(dis_h, .5):.0f} / {pct(dis_h, .9):.0f}")
            print(f"  aspect w/h p10/p50/p90: {pct(ar_dis, .1):.2f} / {pct(ar_dis, .5):.2f} / {pct(ar_dis, .9):.2f}")
            if dis_area_frac:
                print(f"  area % of frame p10/p50/p90: {100 * pct(dis_area_frac, .1):.1f} / "
                      f"{100 * pct(dis_area_frac, .5):.1f} / {100 * pct(dis_area_frac, .9):.1f}")
            spread = pct(ar_dis, .9) / max(pct(ar_dis, .1), 1e-6)
            if spread > 4:
                warns.append(
                    f"{sp}: disinfection box aspect ratio varies {spread:.1f}x between p10 and p90 "
                    "-> box definition is probably inconsistent (plume only vs whole curtain/arch)"
                )
        if sp == "train" and imgs[FOCUS]:
            share = imgs[FOCUS] / max(n_unique, 1)
            if share < 0.15:
                warns.append(
                    f"train: only {share:.1%} of images contain disinfection; consider "
                    "oversampling (repeat those ids 2-3x in train.txt)"
                )
            if negatives / max(n_unique, 1) < 0.05:
                warns.append(
                    "train: <5% negative frames. Add gate frames with spray OFF / sun glare / "
                    "morning haze and NO disinfection box, to cut false positives."
                )
        print()

    for d in ("annotations_cache", "cache_VOC0712"):
        p = os.path.join(devkit, d)
        if os.path.exists(p):
            warns.append(f"old cache exists: {p}  -> delete it if XMLs/images changed (rm -rf)")

    print("========== RESULT ==========")
    for w in warns:
        print("WARN :", w)
    for e in errors:
        print("ERROR:", e)
    if not warns and not errors:
        print("all good")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
