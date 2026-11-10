# Sakku gate model: YOLOX-M 960x960 training

Experiment file: `exps/example/yolox_voc/yolox_voc_m.py`
Classes (order = class id, must match DeepStream `labels.txt`):

```
0 auto  1 bike  2 car  3 disinfection  4 jcb  5 license_plate
6 person  7 security_guard  8 tractor  9 truck
```

## 0. Environment

```bash
pip install -r requirements.txt
pip install -v -e .            # registers the yolox package
export YOLOX_DATADIR=/path/to/datasets   # folder that contains VOCdevkit/
```

COCO-pretrained YOLOX-M weights: `yolox_m.pth` from the official YOLOX release page.
The 80-class head is skipped automatically on load (shape-mismatch warnings for
`head.cls_preds.*` are expected).

## 1. Make the 75/25 split (by clip, not by frame)

```bash
python tools/split_voc_dataset.py --dedupe near            # dry run (hashing takes ~15-30 min)
python tools/split_voc_dataset.py --dedupe near --write    # writes train.txt, valid.txt
```

* File names are `<camera>_<frame number>` (e.g. `SHABAD_gateview_frame_00510`,
  `UNIT6_gateview_1449`). Frames are grouped per camera into blocks of 150 consecutive
  frame numbers; whole blocks go to train or valid. Check the printed `id -> group`
  examples: the part before `#` must be the camera, the number after it the block.
* `--dedupe near` leaves out frames that are byte-identical copies (frame extraction
  wrote many, e.g. SHABAD 101/102/110 are the same file) and frames that are visually
  ~identical to the previous kept frame of the same camera with every box in the same
  place. Nothing is deleted; dropped ids are listed in `ImageSets/Main/dropped_duplicates.txt`.
* Expect a big drop in image count. That is intended: fewer, more varied images train
  faster and the valid score stops counting copies of training frames.
* Old split files are backed up automatically.

## 2. Check the dataset

```bash
rm -rf $YOLOX_DATADIR/VOCdevkit/annotations_cache $YOLOX_DATADIR/VOCdevkit/cache_VOC0712
python tools/check_voc_dataset.py --input-size 960 --check-images
```

Fix every `ERROR` (unknown labels, train/valid overlap, broken boxes) before training.

## 3. Train

```bash
python tools/train.py -f exps/example/yolox_voc/yolox_voc_m.py \
    -c yolox_m.pth -d 1 -b 8 --fp16 -o
```

* `-b 8` on the 24 GB RTX 5090. OOM -> `-b 6`. Watch `gpu mem` for ~300 iters.
* No `--cache ram` on the full set (~1.5 GB per 1000 images at 960).
* Resume: `python tools/train.py -f exps/example/yolox_voc/yolox_voc_m.py -d 1 -b 8 --fp16 --resume`
* TensorBoard: `tensorboard --logdir YOLOX_outputs/yolox_voc_m/tensorboard`
  (per-class AP under `val_AP50/*` and `val_AP50_95/*`).

Outputs in `YOLOX_outputs/yolox_voc_m/`:

| file | meaning |
|---|---|
| `best_ckpt.pth` | best epoch by weighted per-class AP50:95 (disinfection and license_plate count 2x) |
| `last_mosaic_epoch_ckpt.pth` | weights right before the no-aug phase |
| `latest_ckpt.pth` | last epoch, use for `--resume` |
| `training_report_*/` | HTML report, `per_class_ap.csv`, `best_checkpoint.json`, confusion matrix at score >= 0.3 |

## 4. Export for DeepStream / Triton

```bash
python tools/export_onnx.py -f exps/example/yolox_voc/yolox_voc_m.py \
    -c YOLOX_outputs/yolox_voc_m/best_ckpt.pth --output-name yolox_m_960.onnx
```

DeepStream nvinfer must match training preprocessing:

* `infer-dims=3;960;960`, `num-detected-classes=10`, `labels.txt` in the order above
* `net-scale-factor=1.0`, no offsets (YOLOX takes raw 0-255 pixels)
* `model-color-format=1` (BGR)
* `maintain-aspect-ratio=1`, `symmetric-padding=0` (image top-left, pad right/bottom)

## What changed vs. the previous S run

* 10 classes incl. license_plate; class count read from `voc_classes.py`
* 960x960 input, multiscale 896..1024
* Mixup 1.0 -> 0.2, rotation 10 -> 3 deg, shear 2 -> 1, mosaic prob 0.8
* Fog never applied to samples containing disinfection
* Boxes cropped to < 25% of their area by mosaic/affine are dropped
* VOC eval: stale annotation cache rebuilt automatically, float coords handled
* Best checkpoint chosen on weighted per-class AP, per-class AP logged every eval
* Report: objectness loss shown, confusion/P/R at score >= 0.3 with cross-class matching
