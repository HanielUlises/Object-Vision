"""
core/depth_estimator.py — Apple Depth Pro wrapper for metric depth estimation
"""

from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DepthEstimator:
    """
    Wraps Apple's Depth Pro model to produce per-pixel metric depth maps.

    Install depth_pro from: https://github.com/apple/ml-depth-pro
        pip install git+https://github.com/apple/ml-depth-pro
    """

    def __init__(self, settings: Settings):
        try:
            import depth_pro
            self._depth_pro = depth_pro
        except ImportError:
            raise ImportError(
                "depth_pro package not found.\n"
                "Install with: pip install git+https://github.com/apple/ml-depth-pro"
            )

        ckpt = Path(settings.DEPTH_CHECKPOINT)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Depth Pro checkpoint not found: {ckpt}\n"
                "Download from: https://huggingface.co/apple/DepthPro"
            )

        logger.info(f"Loading Depth Pro checkpoint from {ckpt} on {settings.DEVICE}")
        self.device = settings.DEVICE
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device,
            checkpoint_path=str(ckpt),
        )
        self.model.eval()
        self.roi_half = settings.DEPTH_ROI_HALF

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute a metric depth map (in metres) for a BGR frame.

        Returns:
            depth_map: np.ndarray of shape (H, W), float32, values in metres.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image, _, f_px = self._depth_pro.load_rgb_from_array(rgb)
        image_t = self.transform(image).to(self.device)

        with torch.no_grad():
            pred = self.model(image_t)

        depth_map: np.ndarray = pred["depth"].squeeze().cpu().numpy()
        return depth_map.astype(np.float32)

    def get_distance(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """
        Return the robust (median) depth at the centre of a bounding box.

        Args:
            depth_map: full-frame depth map in metres.
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            Estimated distance in metres; 999.0 if not computable.
        """
        if depth_map is None:
            return 999.0

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        h, w = depth_map.shape
        if not (0 <= cy < h and 0 <= cx < w):
            return 999.0

        r = self.roi_half
        roi = depth_map[
            max(0, cy - r): cy + r + 1,
            max(0, cx - r): cx + r + 1,
        ]
        valid = roi[roi > 0]
        if valid.size == 0:
            return float(depth_map[cy, cx])

        return float(np.median(valid))
