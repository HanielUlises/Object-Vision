"""
config/settings.py — Centralized configuration for Obstacle Alert Cam
All defaults can be overridden via settings.yaml or CLI flags.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class Settings:
    # --- Model paths ---
    YOLO_MODEL: str = "models/yolo11s.pt"
    DEPTH_CHECKPOINT: str = "models/depth_pro.pt"

    # --- Detection ---
    YOLO_CONF_THRESHOLD: float = 0.4
    DANGER_CLASSES: List[str] = field(default_factory=lambda: [
        "person", "car", "bicycle", "motorcycle",
        "dog", "cat", "truck", "bus"
    ])

    # --- Distance ---
    DISTANCE_THRESHOLD: float = 1.8   # meters
    DEPTH_ROI_HALF: int = 15          # px radius for median depth sampling
    DEPTH_INTERVAL: float = 0.4       # seconds between depth re-runs

    # --- Alert ---
    ENABLE_BEEP: bool = True
    BEEP_FREQUENCY: int = 1000        # Hz (Windows only)
    BEEP_DURATION: int = 300          # ms  (Windows only)

    # --- UI ---
    SHOW_FPS: bool = False
    DANGER_COLOR: tuple = (0, 0, 255)    # BGR red
    SAFE_COLOR: tuple = (0, 255, 0)      # BGR green
    OVERLAY_ALPHA: float = 0.25

    # --- Device ---
    DEVICE: str = "auto"              # "auto" | "cuda" | "cpu"

    def __post_init__(self):
        if self.DEVICE == "auto":
            import torch
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def save_yaml(self, path: str):
        import dataclasses
        data = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
