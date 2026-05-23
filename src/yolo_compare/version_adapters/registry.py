from __future__ import annotations

from yolo_compare.config import ModelConfig
from yolo_compare.version_adapters.base import VersionAdapter
from yolo_compare.version_adapters.ultralytics_adapter import UltralyticsAdapter
from yolo_compare.version_adapters.yolov5_adapter import YoloV5Adapter
from yolo_compare.version_adapters.yolov6_adapter import YoloV6Adapter
from yolo_compare.version_adapters.yolov7_adapter import YoloV7Adapter


ADAPTERS: dict[str, type[VersionAdapter]] = {
    "ultralytics": UltralyticsAdapter,
    "yolov5": YoloV5Adapter,
    "yolov6": YoloV6Adapter,
    "yolov7": YoloV7Adapter,
}


def build_adapter(model: ModelConfig) -> VersionAdapter:
    adapter_class = ADAPTERS[model.adapter]
    return adapter_class(model)

