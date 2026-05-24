from __future__ import annotations

from importlib import import_module

from yolo_compare.config import ModelConfig
from yolo_compare.version_adapters.base import VersionAdapter


ADAPTERS: dict[str, tuple[str, str]] = {
    "ultralytics": ("yolo_compare.version_adapters.ultralytics_adapter", "UltralyticsAdapter"),
    "rknn": ("yolo_compare.version_adapters.rknn_adapter", "RKNNAdapter"),
    "yolov5": ("yolo_compare.version_adapters.yolov5_adapter", "YoloV5Adapter"),
    "yolov6": ("yolo_compare.version_adapters.yolov6_adapter", "YoloV6Adapter"),
    "yolov7": ("yolo_compare.version_adapters.yolov7_adapter", "YoloV7Adapter"),
}


def build_adapter(model: ModelConfig) -> VersionAdapter:
    module_name, class_name = ADAPTERS[model.adapter]
    adapter_class = getattr(import_module(module_name), class_name)
    return adapter_class(model)
