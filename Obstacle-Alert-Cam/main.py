import argparse
import sys
import cv2
import time

from config.settings import Settings
from core.detector import ObjectDetector
from core.depth_estimator import DepthEstimator
from ui.renderer import Renderer
from utils.alert import AlertSystem
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Obstacle Alert Camera")
    parser.add_argument("--source", default=0,
                        help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Path to settings YAML file (optional)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override distance threshold in meters")
    parser.add_argument("--no-beep", action="store_true",
                        help="Disable audio alerts")
    parser.add_argument("--show-fps", action="store_true",
                        help="Display FPS counter on frame")
    return parser.parse_args()


def main():
    args = parse_args()

    settings = Settings()
    if args.threshold is not None:
        settings.DISTANCE_THRESHOLD = args.threshold
    if args.no_beep:
        settings.ENABLE_BEEP = False
    settings.SHOW_FPS = args.show_fps

    logger.info("Initializing models...")

    # Init subsystems
    detector = ObjectDetector(settings)
    depth_estimator = DepthEstimator(settings)
    renderer = Renderer(settings)
    alert = AlertSystem(settings)

    # Open video source
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        sys.exit(1)

    logger.info("Obstacle Alert Camera running. Press 'q' to quit.")

    depth_map = None
    last_depth_time = 0.0
    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame not received — end of stream or camera error.")
            break

        now = time.time()

        detections = detector.detect(frame)

        if depth_map is None or (now - last_depth_time) > settings.DEPTH_INTERVAL:
            depth_map = depth_estimator.estimate(frame)
            last_depth_time = now

        enriched = []
        triggered_alert = False
        for det in detections:
            dist = depth_estimator.get_distance(depth_map, det["bbox"])
            det["distance"] = dist
            if dist < settings.DISTANCE_THRESHOLD:
                triggered_alert = True
            enriched.append(det)

        if triggered_alert:
            alert.trigger()

        frame_count += 1
        fps = frame_count / (now - fps_time) if (now - fps_time) > 0 else 0

        output = renderer.draw(frame, enriched, triggered_alert,
                               fps=fps if settings.SHOW_FPS else None)

        cv2.imshow("Obstacle Alert", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Session ended.")


if __name__ == "__main__":
    main()
