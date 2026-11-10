#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Leak-free train/valid split for the VOC dataset.

Consecutive CCTV frames are near-identical. A random per-frame split puts copies of
training frames into valid and inflates mAP. This script groups frames into clips
(by file name) and assigns WHOLE groups to train or valid, while keeping every class
close to the requested valid ratio (default 25%).

Grouping: the group key is taken from the image id with --group-regex (first capture
group). Default strips a trailing frame number:
    cam3_20261023_1623_000145   -> cam3_20261023_1623
    gate1-disinf-clip07-frame88 -> gate1-disinf-clip07-frame
If your names look different, pass your own regex and check the printed examples.

Usage:
    python tools/split_voc_dataset.py                    # dry run: prints the split
    python tools/split_voc_dataset.py --write            # writes train.txt / valid.txt
    python tools/split_voc_dataset.py --group-regex '^(cam\\d+_\\d{8})'   # group per camera-day

Old split files are backed up as <name>.txt.bak_<timestamp> before writing.
"""

import argparse
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _load_voc_classes():
    import importlib.util
    p = os.path.join(ROOT, "yolox", "data", "datasets", "voc_classes.py")
    spec = importlib.util.spec_from_file_location("voc_classes", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.VOC_CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.25)
    ap.add_argument("--group-regex", default=r"^(.*?)[_\-]?\d+$",
                    help="regex on image id; capture group 1 = clip key")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-name", default="train")
    ap.add_argument("--valid-name", default="valid")
    ap.add_argument("--write", action="store_true", help="actually write the split files")
    args = ap.parse_args()

    classes = _load_voc_classes()
    if args.voc_root is None:
        base = os.getenv("YOLOX_DATADIR", os.path.join(ROOT, "datasets"))
        args.voc_root = os.path.join(base, "VOCdevkit", "VOC2012")
    ann_dir = os.path.join(args.voc_root, "Annotations")
    img_dir = os.path.join(args.voc_root, "JPEGImages")
    sets_dir = os.path.join(args.voc_root, "ImageSets", "Main")
    pattern = re.compile(args.group_regex)

    # ------------------------------------------------------------ collect
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith(".xml"))
    have_img = {os.path.splitext(f)[0] for f in os.listdir(img_dir)
                if f.lower().endswith(IMG_EXTS)}
    missing = [i for i in ids if i not in have_img]
    ids = [i for i in ids if i in have_img]
    if missing:
        print(f"WARN: {len(missing)} XMLs have no image and are skipped, e.g. {missing[:3]}")

    img_classes = {}
    unknown = Counter()
    for i in ids:
        root = ET.parse(os.path.join(ann_dir, i + ".xml")).getroot()
        names = set()
        for o in root.findall("object"):
            n = (o.findtext("name") or "").strip()
            if n in classes:
                names.add(n)
            else:
                unknown[n] += 1
        img_classes[i] = names
    if unknown:
        print(f"WARN: labels not in VOC_CLASSES (fix before training): {dict(unknown)}")

    groups = defaultdict(list)
    no_match = 0
    for i in ids:
        m = pattern.match(i)
        if m and m.group(1):
            key = m.group(1)
        else:
            key = i
            no_match += 1
        groups[key].append(i)

    print(f"images: {len(ids)}  groups(clips): {len(groups)}  "
          f"ids not matching regex (own group): {no_match}")
    print("example id -> group:")
    for i in random.Random(1).sample(ids, min(8, len(ids))):
        m = pattern.match(i)
        print(f"   {i:<50} -> {m.group(1) if m and m.group(1) else i}")
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"group size: max={sizes[0]}  median={sizes[len(sizes) // 2]}  min={sizes[-1]}")
    if len(groups) < 30:
        print("WARN: very few groups -> the split will be coarse. Check --group-regex.")
    if len(groups) > 0.5 * len(ids):
        print("WARN: most groups have 1-2 frames -> regex is probably not finding clips; "
              "this would be (almost) a per-frame split again. Check --group-regex.")
    if sizes[0] > 0.1 * len(ids):
        print("WARN: one group holds >10% of all images; it will land entirely on one side.")

    # ------------------------------------------------------------ assign
    total_img = Counter()
    for i in ids:
        for c in img_classes[i]:
            total_img[c] += 1
    n_total = len(ids)
    r = args.val_ratio

    ginfo = []
    for k, members in groups.items():
        cnt = Counter()
        for i in members:
            for c in img_classes[i]:
                cnt[c] += 1
        rarity = max((1.0 / total_img[c] for c in cnt), default=0.0)
        ginfo.append((k, members, cnt, rarity))
    import numpy as np

    cls_idx = {c: j for j, c in enumerate(classes)}
    G = len(ginfo)
    gcnt = np.zeros((G, len(classes)))
    gsize = np.zeros(G)
    for gi, (k, members, cnt, _) in enumerate(ginfo):
        gsize[gi] = len(members)
        for c, v in cnt.items():
            gcnt[gi, cls_idx[c]] = v
    tot = np.array([total_img[c] for c in classes], dtype=float)
    present = tot > 0
    # rarer classes count more: a 10-point miss on jcb matters more than on person
    w = np.where(present, 1.0, 0.0)

    def cost_of(vc, nv):
        frac = np.where(present, vc / np.maximum(tot, 1), r)
        return 4.0 * (nv / n_total - r) ** 2 + float(np.sum(w * (frac - r) ** 2))

    best = None
    for trial in range(30):
        rnd = random.Random(args.seed * 1000 + trial)
        order = list(range(G))
        rnd.shuffle(order)
        order.sort(key=lambda gi: -ginfo[gi][3] if trial % 2 == 0 else 0)
        in_val = np.zeros(G, dtype=bool)
        vc = np.zeros(len(classes))
        nv = 0.0
        # greedy pass
        for gi in order:
            if cost_of(vc + gcnt[gi], nv + gsize[gi]) < cost_of(vc, nv):
                in_val[gi] = True
                vc += gcnt[gi]
                nv += gsize[gi]
        # local search: flip single groups / swap pairs while it helps
        cur = cost_of(vc, nv)
        for _ in range(100):
            improved = False
            for gi in rnd.sample(range(G), G):
                sgn = -1.0 if in_val[gi] else 1.0
                c2 = cost_of(vc + sgn * gcnt[gi], nv + sgn * gsize[gi])
                if c2 < cur - 1e-12:
                    in_val[gi] = not in_val[gi]
                    vc += sgn * gcnt[gi]
                    nv += sgn * gsize[gi]
                    cur = c2
                    improved = True
            vi = np.where(in_val)[0]
            ti = np.where(~in_val)[0]
            if len(vi) and len(ti):
                for _ in range(min(2000, len(vi) * len(ti))):
                    a_, b_ = vi[rnd.randrange(len(vi))], ti[rnd.randrange(len(ti))]
                    if in_val[a_] == in_val[b_]:
                        continue
                    vc2 = vc - gcnt[a_] + gcnt[b_]
                    nv2 = nv - gsize[a_] + gsize[b_]
                    c2 = cost_of(vc2, nv2)
                    if c2 < cur - 1e-12:
                        in_val[a_], in_val[b_] = False, True
                        vc, nv, cur = vc2, nv2, c2
                        improved = True
            if not improved:
                break
        if best is None or cur < best[0]:
            best = (cur, in_val.copy(), vc.copy())
    _, in_val, vcv = best
    split = {ginfo[gi][0]: ("valid" if in_val[gi] else "train") for gi in range(G)}
    val_img = Counter({c: int(vcv[cls_idx[c]]) for c in classes})

    train_ids = sorted(i for k, m in groups.items() if split[k] == "train" for i in m)
    valid_ids = sorted(i for k, m in groups.items() if split[k] == "valid" for i in m)

    # ------------------------------------------------------------ report
    print(f"\ntrain: {len(train_ids)}  valid: {len(valid_ids)}  "
          f"(valid = {len(valid_ids) / max(n_total, 1):.1%}, target {r:.0%})")
    print(f"{'class':<16}{'train_img':>10}{'valid_img':>10}{'valid%':>8}")
    bad = []
    for c in classes:
        t = total_img[c] - val_img[c]
        v = val_img[c]
        frac = v / max(total_img[c], 1)
        flag = ""
        if total_img[c] and (frac < r - 0.08 or frac > r + 0.08):
            flag = "  <-- off target (few clips contain this class)"
            bad.append(c)
        print(f"{c:<16}{t:>10}{v:>10}{frac:>8.1%}{flag}")
    neg_t = sum(1 for i in train_ids if not img_classes[i])
    neg_v = sum(1 for i in valid_ids if not img_classes[i])
    print(f"{'negative':<16}{neg_t:>10}{neg_v:>10}")
    leak = set(train_ids) & set(valid_ids)
    assert not leak
    print("groups shared between train and valid: 0")

    if not args.write:
        print("\nDry run. Re-run with --write to save the split.")
        return

    os.makedirs(sets_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, lst in ((args.train_name, train_ids), (args.valid_name, valid_ids)):
        p = os.path.join(sets_dir, name + ".txt")
        if os.path.exists(p):
            shutil.copy2(p, p + f".bak_{stamp}")
        with open(p, "w") as f:
            f.write("\n".join(lst) + "\n")
        print(f"wrote {p}")
    # group map, so the split can be audited / reproduced
    with open(os.path.join(sets_dir, f"split_groups_{stamp}.tsv"), "w") as f:
        for k, members in sorted(groups.items()):
            f.write(f"{k}\t{split[k]}\t{len(members)}\n")
    devkit = os.path.dirname(os.path.abspath(args.voc_root))
    for d in ("annotations_cache", "cache_VOC0712"):
        p = os.path.join(devkit, d)
        if os.path.exists(p):
            print(f"NOTE: delete old cache before training: rm -rf {p}")
    if bad:
        print(f"NOTE: classes off target: {bad}. Usually fine; check valid still has "
              "enough samples of them.")


if __name__ == "__main__":
    sys.exit(main())
