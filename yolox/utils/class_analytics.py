#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""VOC class distribution analytics for training overview reports."""

from __future__ import annotations

import csv
import glob
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NEGATIVE_CLASS_NAME = "negative_sample"


def _read_split_ids(split_file: str) -> set:
    if not os.path.isfile(split_file):
        return set()
    with open(split_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _stem_from_xml(root: ET.Element, xml_path: str) -> str:
    filename = root.findtext("filename")
    if filename:
        return os.path.splitext(filename)[0]
    return os.path.splitext(os.path.basename(xml_path))[0]


def analyze_voc_classes(
    voc_root: str,
    class_names: Sequence[str],
    train_split: str = "train",
    valid_split: str = "valid",
) -> Tuple[List[Dict], Dict]:
    """
    Analyze VOC annotations and return per-class rows + summary.

    Also counts negative samples (XML with zero <object> tags).

    Expected layout:
      <voc_root>/Annotations/*.xml
      <voc_root>/ImageSets/Main/<train_split>.txt
      <voc_root>/ImageSets/Main/<valid_split>.txt
    """
    ann_dir = os.path.join(voc_root, "Annotations")
    sets_dir = os.path.join(voc_root, "ImageSets", "Main")
    train_ids = _read_split_ids(os.path.join(sets_dir, f"{train_split}.txt"))
    valid_ids = _read_split_ids(os.path.join(sets_dir, f"{valid_split}.txt"))

    img_total = Counter()
    box_total = Counter()
    split_img = defaultdict(Counter)
    split_box = defaultdict(Counter)
    unknown_labels = Counter()
    xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))

    negative_total = 0
    negative_split = Counter()
    positive_total = 0
    other_unique = 0

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        stem = _stem_from_xml(root, xml_path)
        names = [obj.findtext("name") for obj in root.findall("object")]
        names = [n for n in names if n]

        if stem in train_ids:
            split = "train"
        elif stem in valid_ids:
            split = "valid"
        else:
            split = "other"
            other_unique += 1

        # Negative sample: annotation exists but no objects
        if len(names) == 0:
            negative_total += 1
            negative_split[split] += 1
            continue

        positive_total += 1
        for name in names:
            box_total[name] += 1
            if name not in class_names:
                unknown_labels[name] += 1

        for name in set(names):
            img_total[name] += 1

        for name in names:
            split_box[split][name] += 1
        for name in set(names):
            split_img[split][name] += 1

    rows = []
    for cls in class_names:
        rows.append(
            {
                "class": cls,
                "images": img_total[cls],
                "boxes": box_total[cls],
                "train_img": split_img["train"][cls],
                "train_box": split_box["train"][cls],
                "valid_img": split_img["valid"][cls],
                "valid_box": split_box["valid"][cls],
            }
        )

    # Extra row for negative / background frames (no bbox)
    rows.append(
        {
            "class": NEGATIVE_CLASS_NAME,
            "images": negative_total,
            "boxes": 0,
            "train_img": int(negative_split["train"]),
            "train_box": 0,
            "valid_img": int(negative_split["valid"]),
            "valid_box": 0,
        }
    )

    summary = {
        "num_xml": len(xml_files),
        "num_train_ids": len(train_ids),
        "num_valid_ids": len(valid_ids),
        "positive_images": positive_total,
        "negative_samples": {
            "total": negative_total,
            "train": int(negative_split["train"]),
            "valid": int(negative_split["valid"]),
            "other": int(negative_split["other"]),
            "train_ratio": round(
                negative_split["train"] / max(len(train_ids), 1), 4
            ),
            "valid_ratio": round(
                negative_split["valid"] / max(len(valid_ids), 1), 4
            ),
        },
        "unknown_labels": dict(unknown_labels),
        "other_split_images": other_unique,
    }
    return rows, summary


def save_class_analytics_csv(
    rows: Iterable[Dict],
    output_csv: str,
) -> str:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "class",
        "images",
        "boxes",
        "train_img",
        "train_box",
        "valid_img",
        "valid_box",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_csv


def generate_class_analytics_report(
    data_dir: Optional[str] = None,
    year: str = "2012",
    class_names: Optional[Sequence[str]] = None,
    train_split: str = "train",
    valid_split: str = "valid",
    output_dir: Optional[str] = None,
) -> str:
    """
    Build class overview CSV under:
      <datasets>/Class-Analysics/class_overview_YYYYMMDD_HHMMSS.csv
    """
    from yolox.data import get_yolox_datadir
    from yolox.data.datasets.voc_classes import VOC_CLASSES

    if data_dir is None:
        data_dir = get_yolox_datadir()
    if class_names is None:
        class_names = VOC_CLASSES

    voc_root = os.path.join(data_dir, "VOCdevkit", f"VOC{year}")
    if output_dir is None:
        output_dir = os.path.join(data_dir, "Class-Analysics")

    rows, summary = analyze_voc_classes(
        voc_root=voc_root,
        class_names=class_names,
        train_split=train_split,
        valid_split=valid_split,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"class_overview_{stamp}.csv")
    save_class_analytics_csv(rows, output_csv)

    # Also keep a stable latest pointer for quick inspection
    latest_csv = os.path.join(output_dir, "class_overview_latest.csv")
    save_class_analytics_csv(rows, latest_csv)

    neg = summary["negative_samples"]
    # Small text summary next to CSV
    summary_txt = os.path.join(output_dir, f"class_overview_{stamp}_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"VOC root: {voc_root}\n")
        f.write(f"XML files: {summary['num_xml']}\n")
        f.write(f"Train IDs ({train_split}): {summary['num_train_ids']}\n")
        f.write(f"Valid IDs ({valid_split}): {summary['num_valid_ids']}\n")
        f.write(f"Positive images (with boxes): {summary['positive_images']}\n")
        f.write(
            "Negative samples (no bbox): "
            f"total={neg['total']}, train={neg['train']}, "
            f"valid={neg['valid']}, other={neg['other']}\n"
        )
        f.write(
            "Negative ratio: "
            f"train={neg['train_ratio']:.2%}, valid={neg['valid_ratio']:.2%}\n"
        )
        f.write(f"Other (not in splits): {summary['other_split_images']}\n")
        if summary["unknown_labels"]:
            f.write(
                f"Unknown labels (not in VOC_CLASSES): {summary['unknown_labels']}\n"
            )
        f.write("\nPer-class:\n")
        f.write(
            f"{'class':<18}{'images':>8}{'boxes':>8}"
            f"{'train_img':>10}{'train_box':>10}"
            f"{'valid_img':>10}{'valid_box':>10}\n"
        )
        for row in rows:
            f.write(
                f"{row['class']:<18}{row['images']:>8}{row['boxes']:>8}"
                f"{row['train_img']:>10}{row['train_box']:>10}"
                f"{row['valid_img']:>10}{row['valid_box']:>10}\n"
            )

    return output_csv
