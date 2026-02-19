"""
ui/renderer.py — Draws bounding boxes, labels, danger overlay, and FPS on frames
"""

from typing import List, Dict, Any, Optional

import cv2
import numpy as np

from config.settings import Settings


class Renderer:
    """Handles all OpenCV drawing operations."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, settings: Settings):
        self.threshold = settings.DISTANCE_THRESHOLD
        self.danger_color = settings.DANGER_COLOR
        self.safe_color = settings.SAFE_COLOR
        self.alpha = settings.OVERLAY_ALPHA

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        alert_active: bool,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render all overlays onto a copy of `frame`.

        Args:
            frame: raw BGR frame.
            detections: list of detection dicts (with 'bbox', 'class_name',
                        'confidence', 'distance').
            alert_active: whether a danger alert is active.
            fps: if not None, drawn in top-left corner.

        Returns:
            Annotated BGR frame.
        """
        out = frame.copy()

        for det in detections:
            self._draw_detection(out, det)

        if alert_active:
            self._draw_danger_overlay(out)

        if fps is not None:
            self._draw_fps(out, fps)

        return out

    def _draw_detection(self, frame: np.ndarray, det: Dict[str, Any]):
        x1, y1, x2, y2 = det["bbox"]
        dist = det.get("distance", 999.0)
        cls = det["class_name"]
        conf = det["confidence"]

        is_danger = dist < self.threshold
        color = self.danger_color if is_danger else self.safe_color

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} {dist:.1f}m ({conf:.0%})"
        (lw, lh), baseline = cv2.getTextSize(label, self.FONT, 0.6, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - lh - baseline - 6),
            (x1 + lw + 2, y1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - baseline - 2),
            self.FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )

        if is_danger:
            cv2.putText(
                frame, "⚠",
                (x2 - 20, y1 + 20),
                self.FONT, 0.7, self.danger_color, 2, cv2.LINE_AA,
            )

    def _draw_danger_overlay(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), self.danger_color, cv2.FILLED)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        banner_h = 70
        cv2.rectangle(frame, (0, 0), (w, banner_h), (0, 0, 180), cv2.FILLED)
        cv2.putText(
            frame, "DANGER — OBSTACLE TOO CLOSE!",
            (30, 50),
            self.FONT, 1.2, (255, 255, 255), 3, cv2.LINE_AA,
        )

    def _draw_fps(self, frame: np.ndarray, fps: float):
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (10, frame.shape[0] - 10),
            self.FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )
