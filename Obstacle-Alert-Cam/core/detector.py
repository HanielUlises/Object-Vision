"""
core/detector.py — YOLO-based object detection wrapper
"""

from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ObjectDetector:
    """Wraps YOLOv8/v11 for real-time object detection."""

    def __init__(self, settings: Settings):
        model_path = Path(settings.YOLO_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}\n"
                "Download with: yolo export model=yolo11s.pt  or place manually."
            )
        logger.info(f"Loading YOLO model from {model_path} on {settings.DEVICE}")
        self.model = YOLO(str(model_path))
        self.conf = settings.YOLO_CONF_THRESHOLD
        self.danger_classes = set(settings.DANGER_CLASSES)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a BGR frame.

        Returns:
            List of dicts with keys:
                - bbox: (x1, y1, x2, y2) ints
                - class_name: str
                - confidence: float
                - distance: float  (filled later by DepthEstimator)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, conf=self.conf, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            if cls_name not in self.danger_classes:
                continue
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": cls_name,
                "confidence": conf,
                "distance": None,
            })

        return detections
