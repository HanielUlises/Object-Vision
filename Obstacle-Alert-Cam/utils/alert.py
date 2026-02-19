"""
utils/alert.py — Cross-platform audio alert system (Windows / macOS / Linux)
"""

import sys
import time
import threading

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class AlertSystem:
    """
    Non-blocking audio alert with cooldown to avoid beep spam.

    Automatically selects the right backend:
        - Windows : winsound.Beep
        - macOS   : afplay system sound
        - Linux   : paplay / aplay / beep (whichever is available)
    """

    COOLDOWN = 0.8

    def __init__(self, settings: Settings):
        self.enabled = settings.ENABLE_BEEP
        self.freq = settings.BEEP_FREQUENCY
        self.duration_ms = settings.BEEP_DURATION
        self._last_beep: float = 0.0
        self._lock = threading.Lock()
        self._platform = sys.platform

    def trigger(self):
        """Fire a beep if cooldown has elapsed (non-blocking)."""
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            if now - self._last_beep < self.COOLDOWN:
                return
            self._last_beep = now
        t = threading.Thread(target=self._beep, daemon=True)
        t.start()

    def _beep(self):
        try:
            if self._platform == "win32":
                import winsound
                winsound.Beep(self.freq, self.duration_ms)
            elif self._platform == "darwin":
                import subprocess
                subprocess.run(
                    ["afplay", "/System/Library/Sounds/Ping.aiff"],
                    check=False,
                )
            else:
                self._linux_beep()
        except Exception as e:
            logger.debug(f"Beep failed: {e}")

    def _linux_beep(self):
        import subprocess, shutil

        dur_s = self.duration_ms / 1000.0

        if shutil.which("paplay"):
            subprocess.run(
                ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                check=False,
            )
        elif shutil.which("aplay"):
            subprocess.run(
                ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
                check=False,
            )
        elif shutil.which("beep"):
            subprocess.run(
                ["beep", "-f", str(self.freq), "-l", str(self.duration_ms)],
                check=False,
            )
        else:
            print("\a", end="", flush=True)
