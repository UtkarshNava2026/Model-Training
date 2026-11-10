#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# Index order here == class id order in the trained model / ONNX / TensorRT engine.
# Keep DeepStream labels.txt and any class-id mapping in the pipeline in this exact order.
# VOC_CLASSES = ( '__background__', # always index 0
VOC_CLASSES = (
    "auto",            # 0
    "bike",            # 1
    "car",             # 2
    "disinfection",    # 3
    "jcb",             # 4
    "license_plate",   # 5
    "person",          # 6
    "security_guard",  # 7
    "tractor",         # 8
    "truck",           # 9
)
