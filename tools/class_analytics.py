#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Generate VOC class distribution overview CSV before training.

Usage:
  python tools/class_analytics.py
  python tools/class_analytics.py --year 2012 --train-split train --valid-split valid
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX class analytics")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root (default: YOLOX/datasets or YOLOX_DATADIR)",
    )
    parser.add_argument("--year", default="2012", help="VOC year folder, e.g. 2012")
    parser.add_argument("--train-split", default="train", help="train split name")
    parser.add_argument("--valid-split", default="valid", help="valid split name")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder (default: <data-dir>/Class-Analysics)",
    )
    return parser


def main():
    args = make_parser().parse_args()
    from yolox.utils.class_analytics import generate_class_analytics_report

    csv_path = generate_class_analytics_report(
        data_dir=args.data_dir,
        year=args.year,
        train_split=args.train_split,
        valid_split=args.valid_split,
        output_dir=args.output_dir,
    )
    print(f"Class analytics saved to: {csv_path}")
    print(
        f"Latest copy: {os.path.join(os.path.dirname(csv_path), 'class_overview_latest.csv')}"
    )


if __name__ == "__main__":
    main()
