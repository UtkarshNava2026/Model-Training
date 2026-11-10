#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Training report package writer.

Creates:
  training_report_YYYYMMDD_HHMMSS/
    training_log.csv
    class_analysis.csv
    dataset_summary.json
    lr_curve.png / loss_curve.png / ap_curve.png
    confusion_matrix.png / pr_curve.png / dataset_statistics.png
    training_config.yaml
    gpu_usage.csv
    model_info.txt
    final_report.html
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    # Avoid home-directory permission issues on some machines
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _bbox_iou(box, boxes):
    """box: [x1,y1,x2,y2], boxes: Nx4"""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area1 = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area1 + area2 - inter + 1e-6
    return inter / union


class TrainingReport:
    def __init__(self, output_root: str, exp, args):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join(output_root, f"training_report_{stamp}")
        os.makedirs(self.report_dir, exist_ok=True)
        self.exp = exp
        self.args = args
        self.class_names = []
        self.epoch_rows: List[Dict[str, Any]] = []
        self.gpu_rows: List[Dict[str, Any]] = []
        self._epoch_loss_sums = defaultdict(float)
        self._epoch_loss_counts = defaultdict(int)
        self._last_predictions = None
        self._last_eval_summary = ""
        self.per_class_rows: List[Dict[str, Any]] = []
        self.best_ap = 0.0
        self.best_ckpt_info: Dict[str, Any] = {
            "checkpoint": "best_ckpt.pth",
            "epoch": None,
            "ap50": None,
            "ap50_95": None,
            "selection_metric": "ap50_95",
            "updated_at": None,
            "checkpoint_path": None,
        }

        self.paths = {
            "training_log": os.path.join(self.report_dir, "training_log.csv"),
            "class_analysis": os.path.join(self.report_dir, "class_analysis.csv"),
            "dataset_summary": os.path.join(self.report_dir, "dataset_summary.json"),
            "best_checkpoint": os.path.join(self.report_dir, "best_checkpoint.json"),
            "per_class_ap": os.path.join(self.report_dir, "per_class_ap.csv"),
            "lr_curve": os.path.join(self.report_dir, "lr_curve.png"),
            "loss_curve": os.path.join(self.report_dir, "loss_curve.png"),
            "ap_curve": os.path.join(self.report_dir, "ap_curve.png"),
            "confusion_matrix": os.path.join(self.report_dir, "confusion_matrix.png"),
            "pr_curve": os.path.join(self.report_dir, "pr_curve.png"),
            "dataset_statistics": os.path.join(self.report_dir, "dataset_statistics.png"),
            "training_config": os.path.join(self.report_dir, "training_config.yaml"),
            "gpu_usage": os.path.join(self.report_dir, "gpu_usage.csv"),
            "model_info": os.path.join(self.report_dir, "model_info.txt"),
            "final_report": os.path.join(self.report_dir, "final_report.html"),
        }

    # ------------------------------------------------------------------ start
    def start(self, model_info: str = ""):
        from yolox.data.datasets.voc_classes import VOC_CLASSES
        from yolox.utils.class_analytics import (
            analyze_voc_classes,
            generate_class_analytics_report,
            save_class_analytics_csv,
        )
        from yolox.data import get_yolox_datadir

        self.class_names = list(VOC_CLASSES)

        # class analysis into report folder + global Class-Analysics
        try:
            src_csv = generate_class_analytics_report()
            if os.path.isfile(src_csv):
                shutil.copy2(src_csv, self.paths["class_analysis"])
            latest = os.path.join(
                os.path.dirname(src_csv), "class_overview_latest.csv"
            )
            if os.path.isfile(latest) and not os.path.isfile(self.paths["class_analysis"]):
                shutil.copy2(latest, self.paths["class_analysis"])
        except Exception:
            # fallback direct analyze
            voc_root = os.path.join(get_yolox_datadir(), "VOCdevkit", "VOC2012")
            rows, summary = analyze_voc_classes(voc_root, self.class_names)
            save_class_analytics_csv(rows, self.paths["class_analysis"])
            with open(self.paths["dataset_summary"], "w") as f:
                json.dump(summary, f, indent=2)

        # dataset_summary.json
        try:
            voc_root = os.path.join(get_yolox_datadir(), "VOCdevkit", "VOC2012")
            rows, summary = analyze_voc_classes(voc_root, self.class_names)
            summary_out = {
                "voc_root": voc_root,
                "num_classes": len(self.class_names),
                "class_names": self.class_names,
                "splits": summary,
                "per_class": rows,
            }
            with open(self.paths["dataset_summary"], "w") as f:
                json.dump(summary_out, f, indent=2)
            self._plot_dataset_statistics(rows)
        except Exception as e:
            with open(self.paths["dataset_summary"], "w") as f:
                json.dump({"error": str(e)}, f, indent=2)

        self._write_training_config()
        with open(self.paths["model_info"], "w") as f:
            f.write(model_info.strip() + "\n" if model_info else "N/A\n")
            f.write(f"\nnum_classes: {getattr(self.exp, 'num_classes', 'N/A')}\n")
            f.write(f"input_size: {getattr(self.exp, 'input_size', 'N/A')}\n")
            f.write(f"test_size: {getattr(self.exp, 'test_size', 'N/A')}\n")
            f.write(f"max_epoch: {getattr(self.exp, 'max_epoch', 'N/A')}\n")

        # init empty csvs
        self._write_training_log()
        self._write_gpu_usage()

    def _write_training_config(self):
        cfg = {
            "experiment_name": getattr(self.args, "experiment_name", None),
            "exp_file": getattr(self.args, "exp_file", None),
            "batch_size": getattr(self.args, "batch_size", None),
            "fp16": getattr(self.args, "fp16", None),
            "ckpt": getattr(self.args, "ckpt", None),
            "devices": getattr(self.args, "devices", None),
            "exp": {},
        }
        for k, v in sorted(vars(self.exp).items()):
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool, list, tuple, dict)) or v is None:
                try:
                    json.dumps(v)
                    cfg["exp"][k] = v if not isinstance(v, tuple) else list(v)
                except Exception:
                    cfg["exp"][k] = str(v)

        if yaml is not None:
            with open(self.paths["training_config"], "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        else:
            with open(self.paths["training_config"], "w") as f:
                f.write(json.dumps(cfg, indent=2))

    # -------------------------------------------------------------- logging
    def log_iter(
        self,
        epoch: int,
        iteration: int,
        lr: float,
        losses: Dict[str, float],
        gpu_mb: float,
        mem_gb: float,
        iter_time: float,
    ):
        self.gpu_rows.append(
            {
                "epoch": epoch,
                "iter": iteration,
                "gpu_mem_mb": round(_safe_float(gpu_mb), 2),
                "host_mem_gb": round(_safe_float(mem_gb), 3),
                "lr": _safe_float(lr),
                "iter_time_s": round(_safe_float(iter_time), 4),
            }
        )
        for k, v in losses.items():
            self._epoch_loss_sums[k] += _safe_float(v)
            self._epoch_loss_counts[k] += 1
        # flush gpu csv periodically
        if len(self.gpu_rows) % 20 == 0:
            self._write_gpu_usage()

    def end_epoch(
        self,
        epoch: int,
        lr: float,
        fps: float = 0.0,
        gpu_mb: float = 0.0,
        eta: str = "",
        ap50: Optional[float] = None,
        ap50_95: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
    ):
        def avg(key, alt_keys=None):
            if self._epoch_loss_counts.get(key, 0) > 0:
                return self._epoch_loss_sums[key] / self._epoch_loss_counts[key]
            if alt_keys:
                for k in alt_keys:
                    if self._epoch_loss_counts.get(k, 0) > 0:
                        return self._epoch_loss_sums[k] / self._epoch_loss_counts[k]
            return 0.0

        row = {
            "epoch": epoch,
            "train_loss": round(avg("total_loss", ["loss"]), 4),
            "iou_loss": round(avg("iou_loss"), 4),
            # YOLOX logs objectness loss as "conf_loss"
            "obj_loss": round(avg("obj_loss", ["conf_loss"]), 4),
            "cls_loss": round(avg("cls_loss"), 4),
            "lr": round(_safe_float(lr), 6),
            "ap50": "" if ap50 is None else round(_safe_float(ap50) * 100.0, 2),
            "ap50_95": "" if ap50_95 is None else round(_safe_float(ap50_95) * 100.0, 2),
            "precision": "" if precision is None else round(_safe_float(precision), 4),
            "recall": "" if recall is None else round(_safe_float(recall), 4),
            "fps": round(_safe_float(fps), 2),
            "gpu_mem_gb": round(_safe_float(gpu_mb) / 1024.0, 2),
            "eta": eta,
        }
        # update existing epoch row if present
        replaced = False
        for i, old in enumerate(self.epoch_rows):
            if old["epoch"] == epoch:
                # keep previous AP if this call has none
                if ap50 is None and old.get("ap50") != "":
                    row["ap50"] = old["ap50"]
                    row["ap50_95"] = old["ap50_95"]
                    row["precision"] = old.get("precision", "")
                    row["recall"] = old.get("recall", "")
                self.epoch_rows[i] = row
                replaced = True
                break
        if not replaced:
            self.epoch_rows.append(row)

        self._epoch_loss_sums.clear()
        self._epoch_loss_counts.clear()
        self._write_training_log()
        self._write_gpu_usage()
        self._plot_curves()

    def log_eval(
        self,
        epoch: int,
        ap50: float,
        ap50_95: float,
        summary: str = "",
        predictions=None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        is_best: bool = False,
        ckpt_dir: Optional[str] = None,
        per_class_ap: Optional[Dict[str, Dict[str, float]]] = None,
        selection_metric: str = "ap50_95",
        selection_score: Optional[float] = None,
    ):
        ap50_f = _safe_float(ap50)
        if per_class_ap:
            self.per_class_rows.append({"epoch": int(epoch), **{
                f"{c}_{k}": round(100.0 * v[k], 2)
                for c, v in per_class_ap.items() for k in ("ap50", "ap50_95")
            }})
            self._write_per_class_ap()
        ap50_95_f = _safe_float(ap50_95)
        self.best_ap = max(self.best_ap, ap50_95_f)
        self._last_eval_summary = summary or ""
        if predictions is not None:
            self._last_predictions = predictions

        # Update AP fields on existing epoch row (do not recompute losses)
        updated = False
        for row in self.epoch_rows:
            if row["epoch"] == epoch:
                row["ap50"] = round(ap50_f * 100.0, 2)
                row["ap50_95"] = round(ap50_95_f * 100.0, 2)
                if precision is not None:
                    row["precision"] = round(_safe_float(precision), 4)
                if recall is not None:
                    row["recall"] = round(_safe_float(recall), 4)
                updated = True
                break
        if not updated:
            self.epoch_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": 0.0,
                    "iou_loss": 0.0,
                    "obj_loss": 0.0,
                    "cls_loss": 0.0,
                    "lr": 0.0,
                    "ap50": round(ap50_f * 100.0, 2),
                    "ap50_95": round(ap50_95_f * 100.0, 2),
                    "precision": ""
                    if precision is None
                    else round(_safe_float(precision), 4),
                    "recall": "" if recall is None else round(_safe_float(recall), 4),
                    "fps": 0.0,
                    "gpu_mem_gb": 0.0,
                    "eta": "",
                }
            )

        # Track metrics linked to best_ckpt.pth (selected by ap50_95)
        if is_best or self.best_ckpt_info["epoch"] is None:
            ckpt_path = None
            if ckpt_dir:
                ckpt_path = os.path.join(ckpt_dir, "best_ckpt.pth")
            self.best_ckpt_info = {
                "checkpoint": "best_ckpt.pth",
                "checkpoint_path": ckpt_path,
                "epoch": int(epoch),
                "ap50": round(ap50_f * 100.0, 2),
                "ap50_95": round(ap50_95_f * 100.0, 2),
                "selection_metric": selection_metric,
                "selection_score": None if selection_score is None
                else round(100.0 * selection_score, 2),
                "per_class_ap": None if not per_class_ap else {
                    c: {k: round(100.0 * v[k], 2) for k in ("ap50", "ap50_95")}
                    for c, v in per_class_ap.items()
                },
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._write_best_checkpoint()

        self._write_training_log()
        self._plot_curves()

    def _write_best_checkpoint(self):
        # Also keep max from log as fallback if is_best never flagged
        info = dict(self.best_ckpt_info)
        with open(self.paths["best_checkpoint"], "w") as f:
            json.dump(info, f, indent=2)

    # --------------------------------------------------------------- writers
    def _write_training_log(self):
        fields = [
            "epoch",
            "train_loss",
            "iou_loss",
            "obj_loss",
            "cls_loss",
            "lr",
            "ap50",
            "ap50_95",
            "precision",
            "recall",
            "fps",
            "gpu_mem_gb",
            "eta",
        ]
        with open(self.paths["training_log"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in self.epoch_rows:
                w.writerow(row)

    def _write_per_class_ap(self):
        if not self.per_class_rows:
            return
        fields = ["epoch"] + [k for k in self.per_class_rows[-1].keys() if k != "epoch"]
        with open(self.paths["per_class_ap"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in self.per_class_rows:
                w.writerow(row)

    def _write_gpu_usage(self):
        fields = ["epoch", "iter", "gpu_mem_mb", "host_mem_gb", "lr", "iter_time_s"]
        with open(self.paths["gpu_usage"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in self.gpu_rows:
                w.writerow(row)

    # ----------------------------------------------------------------- plots
    def _plot_dataset_statistics(self, rows: List[Dict]):
        if plt is None or not rows:
            return
        from yolox.utils.class_analytics import NEGATIVE_CLASS_NAME

        classes = [r["class"] for r in rows]
        images = [r["images"] for r in rows]
        boxes = [r["boxes"] for r in rows]
        colors = [
            "#d62728" if c == NEGATIVE_CLASS_NAME else "#1f77b4" for c in classes
        ]
        x = np.arange(len(classes))
        width = 0.38
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width / 2, images, width, color=colors, label="Images")
        ax.bar(x + width / 2, boxes, width, color="#ff7f0e", label="Boxes")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=35, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Dataset class statistics (red = negative samples)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.paths["dataset_statistics"], dpi=140)
        plt.close(fig)

    def _plot_curves(self):
        if plt is None or not self.epoch_rows:
            return
        epochs = [r["epoch"] for r in self.epoch_rows]

        # LR
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, [r["lr"] for r in self.epoch_rows], marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title("Learning rate curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.paths["lr_curve"], dpi=140)
        plt.close(fig)

        # Losses
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, label in [
            ("train_loss", "Train Loss"),
            ("iou_loss", "IoU Loss"),
            ("obj_loss", "Obj Loss"),
            ("cls_loss", "Cls Loss"),
        ]:
            ax.plot(epochs, [r[key] for r in self.epoch_rows], marker="o", label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.paths["loss_curve"], dpi=140)
        plt.close(fig)

        # AP
        ap50, ap95, ep = [], [], []
        for r in self.epoch_rows:
            if r["ap50"] == "" or r["ap50_95"] == "":
                continue
            ep.append(r["epoch"])
            ap50.append(r["ap50"])
            ap95.append(r["ap50_95"])
        if ep:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ep, ap50, marker="o", label="AP50")
            ax.plot(ep, ap95, marker="o", label="AP50-95")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("AP (%)")
            ax.set_title("Validation AP curves")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.paths["ap_curve"], dpi=140)
            plt.close(fig)

    def _compute_detection_stats(
        self, predictions, iou_thr: float = 0.5
    ) -> Tuple[np.ndarray, List[float], List[float], float, float]:
        """Confusion matrix + per-class / micro P/R at the deployment score threshold.

        * PR curves use ALL detections (that is what a PR curve is).
        * Confusion matrix, P and R use only detections with score >= exp.report_conf_thre
          (default 0.3). The old version used the eval threshold (0.01), so most
          "false positives" were low-score boxes nobody deploys with.
        * Matching is cross-class: a prediction that hits a GT of ANOTHER class (IoU>=0.5)
          is counted as confusion [pred_cls, gt_cls] instead of background, so e.g.
          person boxes fired on security guards show up as person->security_guard.
        """
        from yolox.data import get_yolox_datadir
        from yolox.data.datasets.voc import VOCDetection
        from yolox.data.data_augment import ValTransform

        num_classes = len(self.class_names)
        conf_thr = float(getattr(self.exp, "report_conf_thre", 0.3))
        self.report_conf_thr = conf_thr
        conf = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)  # + BG
        dataset = VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[("2012", "valid")],
            img_size=getattr(self.exp, "test_size", (640, 640)),
            preproc=ValTransform(legacy=False),
        )

        gt_count = np.zeros(num_classes, dtype=np.int64)
        scores_by_cls = [[] for _ in range(num_classes)]
        matched_by_cls = [[] for _ in range(num_classes)]

        for idx in range(len(dataset)):
            pred = predictions.get(int(idx)) if isinstance(predictions, dict) else None
            if pred is None or pred[0] is None:
                bboxes = np.zeros((0, 4))
                pcls = np.zeros((0,), dtype=int)
                scores = np.zeros((0,))
            else:
                bboxes = np.asarray(pred[0])
                pcls = np.asarray(pred[1]).astype(int)
                scores = np.asarray(pred[2])

            ann, (h0, w0), _ = dataset.annotations[idx]
            if len(ann):
                r = min(float(dataset.img_size[0]) / float(h0),
                        float(dataset.img_size[1]) / float(w0))
                gt_boxes = ann[:, :4] / max(r, 1e-6)
                gt_cls = ann[:, 4].astype(int)
            else:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
                gt_cls = np.zeros((0,), dtype=int)
            for g in gt_cls:
                if 0 <= g < num_classes:
                    gt_count[g] += 1

            order = np.argsort(-scores) if len(scores) else []
            ious = [_bbox_iou(bboxes[pi], gt_boxes) for pi in range(len(scores))]

            # pass 1: PR curve data, same-class greedy matching, all scores
            matched = set()
            for pi in order:
                cp = int(pcls[pi])
                if cp < 0 or cp >= num_classes:
                    continue
                best, best_iou = -1, 0.0
                for gi, v in enumerate(ious[pi]):
                    if gi not in matched and gt_cls[gi] == cp and v > best_iou:
                        best, best_iou = gi, v
                scores_by_cls[cp].append(float(scores[pi]))
                hit = best >= 0 and best_iou >= iou_thr
                if hit:
                    matched.add(best)
                matched_by_cls[cp].append(1 if hit else 0)

            # pass 2: confusion at deployment threshold, cross-class
            matched = set()
            for pi in order:
                cp = int(pcls[pi])
                if cp < 0 or cp >= num_classes or scores[pi] < conf_thr:
                    continue
                best, best_iou = -1, 0.0
                for gi, v in enumerate(ious[pi]):
                    if gi not in matched and gt_cls[gi] == cp and v > best_iou:
                        best, best_iou = gi, v
                if best >= 0 and best_iou >= iou_thr:
                    matched.add(best)
                    conf[cp, cp] += 1
                    continue
                # other-class GT (matched or not: also catches person+guard duplicates)
                ob, ob_iou = -1, 0.0
                for gi, v in enumerate(ious[pi]):
                    if gt_cls[gi] != cp and v > ob_iou:
                        ob, ob_iou = gi, v
                if ob >= 0 and ob_iou >= iou_thr:
                    conf[cp, int(gt_cls[ob])] += 1
                else:
                    conf[cp, num_classes] += 1  # background FP
            for gi in range(len(gt_cls)):
                if gi not in matched and 0 <= gt_cls[gi] < num_classes:
                    conf[num_classes, int(gt_cls[gi])] += 1  # missed (FN)

        precisions, recalls = [], []
        for c in range(num_classes):
            tp = conf[c, c]
            pred_total = conf[c, :].sum()
            precisions.append(tp / (pred_total + 1e-6))
            recalls.append(tp / (gt_count[c] + 1e-6))
        tp_all = float(np.trace(conf[:num_classes, :num_classes]))
        micro_p = tp_all / (conf[:num_classes, :].sum() + 1e-6)
        micro_r = tp_all / (gt_count.sum() + 1e-6)
        self._gt_count = gt_count
        return conf, precisions, recalls, micro_p, micro_r, scores_by_cls, matched_by_cls

    def _plot_confusion_and_pr(self):
        if self._last_predictions is None or plt is None:
            return None, None
        try:
            (
                conf,
                precisions,
                recalls,
                micro_p,
                micro_r,
                scores_by_cls,
                matched_by_cls,
            ) = self._compute_detection_stats(self._last_predictions)
        except Exception:
            return None, None

        labels = list(self.class_names) + ["BG"]
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(conf, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_title("Confusion matrix (IoU=0.5, score >= %.2f)" % self.report_conf_thr)
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                ax.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(self.paths["confusion_matrix"], dpi=140)
        plt.close(fig)

        # PR curve per class from ranked detections
        fig, ax = plt.subplots(figsize=(9, 6))
        plotted = False
        for c, name in enumerate(self.class_names):
            scores = np.asarray(scores_by_cls[c], dtype=np.float32)
            matched = np.asarray(matched_by_cls[c], dtype=np.float32)
            if scores.size == 0:
                continue
            order = np.argsort(-scores)
            matched = matched[order]
            tp = np.cumsum(matched)
            fp = np.cumsum(1.0 - matched)
            # TP + FN as GT count
            gt_count = float(self._gt_count[c])
            if gt_count <= 0:
                continue
            precision = tp / np.maximum(tp + fp, 1e-6)
            recall = tp / gt_count
            ax.plot(
                recall,
                precision,
                label=f"{name} (P={precisions[c]:.2f}, R={recalls[c]:.2f})",
            )
            plotted = True
        if plotted:
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("PR curves (IoU=0.5); P/R in legend at score >= %.2f"
                         % self.report_conf_thr)
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="lower left")
            fig.tight_layout()
            fig.savefig(self.paths["pr_curve"], dpi=140)
        plt.close(fig)
        return micro_p, micro_r

    # --------------------------------------------------------------- finalize
    def finalize(self):
        self._write_training_log()
        self._write_gpu_usage()
        self._plot_curves()
        # Ensure best_checkpoint.json exists even if is_best flag was missed
        if self.best_ckpt_info.get("epoch") is None:
            best_row = None
            best_ap95 = -1.0
            for row in self.epoch_rows:
                if row.get("ap50_95") == "":
                    continue
                val = _safe_float(row["ap50_95"])
                if val > best_ap95:
                    best_ap95 = val
                    best_row = row
            if best_row is not None:
                exp_dir = getattr(self.args, "experiment_name", None)
                out_dir = getattr(self.exp, "output_dir", "YOLOX_outputs")
                ckpt_path = None
                if exp_dir:
                    ckpt_path = os.path.join(out_dir, exp_dir, "best_ckpt.pth")
                self.best_ckpt_info = {
                    "checkpoint": "best_ckpt.pth",
                    "checkpoint_path": ckpt_path,
                    "epoch": int(best_row["epoch"]),
                    "ap50": _safe_float(best_row["ap50"]),
                    "ap50_95": _safe_float(best_row["ap50_95"]),
                    "selection_metric": "ap50_95",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
        self._write_best_checkpoint()
        micro_p, micro_r = self._plot_confusion_and_pr()
        if micro_p is not None and self.epoch_rows:
            self.epoch_rows[-1]["precision"] = round(float(micro_p), 4)
            self.epoch_rows[-1]["recall"] = round(float(micro_r), 4)
            self._write_training_log()
        self._write_html(micro_p, micro_r)
        return self.report_dir

    def _write_html(self, micro_p=None, micro_r=None):
        def img_tag(name, path):
            if os.path.isfile(path):
                src = os.path.basename(path)
                return (
                    f"<h3>{name}</h3>"
                    f'<img src="{src}" '
                    f'style="max-width:100%;border:1px solid #ddd;"/>'
                )
            return f"<h3>{name}</h3><p><i>Not available</i></p>"

        rows_html = ""
        for r in self.epoch_rows:
            rows_html += (
                "<tr>"
                + "".join(f"<td>{r.get(k, '')}</td>" for k in [
                    "epoch", "train_loss", "iou_loss", "obj_loss", "cls_loss",
                    "lr", "ap50", "ap50_95", "precision", "recall", "fps",
                    "gpu_mem_gb", "eta",
                ])
                + "</tr>"
            )

        best = max(
            (_safe_float(r["ap50_95"]) for r in self.epoch_rows if r["ap50_95"] != ""),
            default=0.0,
        )
        b = self.best_ckpt_info
        best_block = (
            f"<h2>Best checkpoint (<code>best_ckpt.pth</code>)</h2>"
            f"<ul>"
            f"<li>Epoch: <b>{b.get('epoch')}</b></li>"
            f"<li>AP50: <b>{b.get('ap50')}</b></li>"
            f"<li>AP50-95: <b>{b.get('ap50_95')}</b></li>"
            f"<li>Selection metric: <code>{b.get('selection_metric')}</code>"
            f" = <b>{b.get('selection_score', b.get('ap50_95'))}</b></li>"
            f"<li>Path: <code>{b.get('checkpoint_path')}</code></li>"
            f"</ul>"
        )
        pc = b.get("per_class_ap") or {}
        if pc:
            best_block += (
                "<h3>Per-class AP of best checkpoint</h3><table>"
                "<tr><th>Class</th><th>AP50</th><th>AP50-95</th></tr>"
                + "".join(
                    f"<tr><td>{c}</td><td>{v['ap50']}</td><td>{v['ap50_95']}</td></tr>"
                    for c, v in pc.items()
                )
                + "</table><p>Per-epoch values: <code>per_class_ap.csv</code></p>"
            )
        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>YOLOX Training Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f3f3f3; }}
    code {{ background: #f6f6f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>YOLOX Training Report</h1>
  <p>Generated: {datetime.now().isoformat(timespec="seconds")}</p>
  <p>Experiment: <code>{getattr(self.args, "experiment_name", "")}</code></p>
  <p>Best AP50-95: <b>{best}</b>
     {
        ("| Final Precision: <b>%.3f</b> | Final Recall: <b>%.3f</b> (score &ge; %.2f)"
         % (micro_p, micro_r, getattr(self, "report_conf_thr", 0.3)))
        if micro_p is not None else ""
     }
  </p>
  {best_block}

  <h2>Training log</h2>
  <table>
    <tr>
      <th>Epoch</th><th>Train Loss</th><th>IoU</th><th>Obj</th><th>Cls</th>
      <th>LR</th><th>AP50</th><th>AP50-95</th><th>Precision</th><th>Recall</th>
      <th>FPS</th><th>GPU Mem</th><th>ETA</th>
    </tr>
    {rows_html}
  </table>

  {img_tag("Dataset statistics", self.paths["dataset_statistics"])}
  {img_tag("Loss curves", self.paths["loss_curve"])}
  {img_tag("Learning rate", self.paths["lr_curve"])}
  {img_tag("AP curves", self.paths["ap_curve"])}
  {img_tag("Confusion matrix", self.paths["confusion_matrix"])}
  {img_tag("PR curves", self.paths["pr_curve"])}

  <h2>Files</h2>
  <ul>
    <li><code>training_log.csv</code></li>
    <li><code>best_checkpoint.json</code></li>
    <li><code>per_class_ap.csv</code></li>
    <li><code>class_analysis.csv</code></li>
    <li><code>dataset_summary.json</code></li>
    <li><code>training_config.yaml</code></li>
    <li><code>gpu_usage.csv</code></li>
    <li><code>model_info.txt</code></li>
  </ul>
</body>
</html>
"""
        with open(self.paths["final_report"], "w") as f:
            f.write(html)
