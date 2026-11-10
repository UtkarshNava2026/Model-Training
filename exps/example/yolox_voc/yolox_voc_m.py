# encoding: utf-8
"""
YOLOX-M on the Sakku gate VOC dataset (10 classes), 960x960, RTX 5090 24 GB.

Full procedure: docs/TRAINING_SAKKU.md

Train (single GPU, from COCO-pretrained YOLOX-M):
    python tools/train.py -f exps/example/yolox_voc/yolox_voc_m.py \
        -c yolox_m.pth -d 1 -b 8 --fp16 -o

  -b 8   : start here on 24 GB. Watch "gpu mem" in the log over the first ~300 iters
           (multiscale goes up to 1024). OOM -> -b 6. Lots of headroom -> -b 10/12.
           LR scales with batch automatically (basic_lr_per_img * batch).
  --cache: each cached image is ~960x540x3 = 1.5 MB -> ~150 GB RAM for 100k images.
           Do NOT use --cache ram on the full set. Use no cache (or --cache disk if
           the disk is fast NVMe and has the space).
  resume : python tools/train.py -f <this file> -d 1 -b 8 --fp16 --resume

Before every run where XMLs / splits changed:
    rm -rf datasets/VOCdevkit/annotations_cache datasets/VOCdevkit/cache_VOC0712
    python tools/check_voc_dataset.py --input-size 960
"""
import os

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model ---------------- #
        from yolox.data.datasets.voc_classes import VOC_CLASSES
        self.num_classes = len(VOC_CLASSES)  # 10
        # YOLOX-M scale
        self.depth = 0.67
        self.width = 0.75

        # ---------------- input ---------------- #
        # 960 for small objects (license plates). DeepStream/TensorRT engine must use
        # the same size: infer-dims=3;960;960
        self.input_size = (960, 960)
        self.test_size = (960, 960)
        # 960 +- 2*32 => 896..1024. Range 5 would reach 1120: memory spikes that can OOM
        # a 24 GB card mid-epoch, for little gain at this resolution.
        self.multiscale_range = 2
        # mosaic at 960 is CPU heavy; raise if the server has free cores and GPU util < 90%
        self.data_num_workers = 8

        # ---------------- schedule ---------------- #
        # ~100k train images at 960 on M is ~3-4 h/epoch on a 24 GB 5090 (estimated from
        # the S@640 run: ~42 min/epoch). 60 epochs ~= 8-10 days. With COCO-pretrained
        # weights and 100k images this is enough: in the S run val AP was flat for the
        # last 40 mosaic epochs and all the late gain came from the no-aug phase, so the
        # no-aug phase is longer here (20) instead of more total epochs.
        self.warmup_epochs = 3
        self.max_epoch = 60
        self.no_aug_epochs = 20
        # eval on ~33k val images at 960 takes a while; every 10 epochs during the mosaic
        # phase. The trainer switches to every epoch for the last no_aug_epochs.
        self.eval_interval = 10
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 5e-4
        self.save_history_ckpt = False  # M ckpts are ~200MB each; keep best/latest/last_epoch

        # ---------------- augmentation ---------------- #
        # Mosaic: keep, but leave ~20% full-frame samples so the model also sees the
        # gate in its real fixed-camera context (spray between the curtains, full plume).
        self.mosaic_prob = 0.8
        self.mosaic_scale = (0.5, 1.5)
        # MixUp blends two frames 50/50. That turns every frame into a "hazy" frame and
        # halves the contrast of the mist itself, while still keeping the disinfection
        # box => noisy labels. Keep it rare.
        self.enable_mixup = True
        self.mixup_prob = 0.2
        self.mixup_scale = (0.8, 1.2)
        # Fixed CCTV: large rotations do not occur at test time and they inflate the
        # axis-aligned boxes of tall/wide objects (disinfection, trucks).
        self.degrees = 3.0
        self.translate = 0.1
        self.shear = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        # Rain/fog: each on ~5% of samples. Fog is NEVER applied to a sample that contains
        # a disinfection box (it washes out the mist but the box stays). Fogged frames
        # without disinfection still teach "global haze != disinfection".
        self.weather_prob = 0.05
        self.no_fog_class_ids = (VOC_CLASSES.index("disinfection"),)
        # Drop boxes that mosaic/affine cropped to <25% of their area. Stops the model
        # learning "a thin sliver of mist / curtain at a tile edge = disinfection".
        self.min_box_area_ratio = 0.25

        # ---------------- eval ---------------- #
        self.test_conf = 0.01   # for mAP only (needs low-score detections)
        self.nmsthre = 0.65
        # confusion matrix / precision / recall in the training report are computed at
        # this score, i.e. roughly what DeepStream uses (pre-cluster-threshold).
        self.report_conf_thre = 0.3
        # best_ckpt.pth = epoch with the highest WEIGHTED mean of per-class AP50:95.
        # Classes not listed weigh 1.0. Plain mAP is still logged.
        self.best_ckpt_class_weights = {"disinfection": 2.0, "license_plate": 2.0}

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import VOCDetection, TrainTransform

        return VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[("2012", "train")],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=80,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                weather_prob=self.weather_prob,
                no_fog_class_ids=self.no_fog_class_ids,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import VOCDetection, ValTransform

        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[("2012", "valid")],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(
                batch_size, is_distributed, testdev=testdev, legacy=legacy
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
