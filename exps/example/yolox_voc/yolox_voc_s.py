# encoding: utf-8
import os

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # Always in sync with yolox/data/datasets/voc_classes.py
        from yolox.data.datasets.voc_classes import VOC_CLASSES
        self.num_classes = len(VOC_CLASSES)

        # YOLOX-S scale
        self.depth = 0.33
        self.width = 0.50

        # S trains at 640 (Nano used 416)
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        # Multiscale around 640: [640-5*32, 640+5*32] => ~480..800
        self.multiscale_range = 5

        self.warmup_epochs = 5
        self.max_epoch = 40
        self.eval_interval = 5
        self.no_aug_epochs = 15

        # ---------- transform config ------------ #
        # Slightly milder than COCO defaults for gate / small objects
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.enable_mixup = True
        self.mosaic_scale = (0.5, 1.5)
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        # Albumentations: heavy rain and fog each on ~10% of samples
        self.weather_prob = 0.1
        # never fog a sample that contains disinfection
        self.no_fog_class_ids = (VOC_CLASSES.index("disinfection"),)

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
