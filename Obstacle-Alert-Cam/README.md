# 🚨 BREAKING: Obstacle Alert Cam

Real-time obstacle detection + metric depth estimation.  
Objects closer than a configurable distance threshold trigger a visual danger overlay and an audio beep.

---

## Project Structure

```
obstacle_alert_cam/
├── main.py                  # Entry point
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── settings.py          # Dataclass-based config
│   └── settings.yaml        # User-editable defaults
├── core/
│   ├── __init__.py
│   ├── detector.py          # YOLO object detection
│   └── depth_estimator.py   # Apple Depth Pro metric depth
├── ui/
│   ├── __init__.py
│   └── renderer.py          # OpenCV drawing / overlays
├── utils/
│   ├── __init__.py
│   ├── alert.py             # Cross-platform audio alert
│   └── logger.py            # Centralised logging
└── models/                  # Place model weights here (gitignored)
    ├── yolo11s.pt
    └── depth_pro.pt
```

---

## Setup

### 1. Clone & create environment

```bash
git clone <your-repo>
cd obstacle_alert_cam
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# Install Depth Pro (not on PyPI)
pip install git+https://github.com/apple/ml-depth-pro
```

### 3. Download model weights

**YOLO** (auto-downloaded on first run by ultralytics, or manually):
```bash
mkdir -p models
# ultralytics will download yolo11s.pt automatically on first use
```

**Depth Pro checkpoint** (~1 GB):
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('apple/DepthPro', 'depth_pro.pt', local_dir='models')
"
```

---

## Usage

```bash
# Default: webcam, default settings
python main.py

# Use a video file
python main.py --source path/to/video.mp4

# Custom distance threshold (metres)
python main.py --threshold 2.5

# Show FPS counter
python main.py --show-fps

# Disable audio
python main.py --no-beep

# Load custom settings file
python main.py --config my_settings.yaml
```

Press **`q`** to quit.

---

## Configuration

Edit `config/settings.yaml` to change defaults without touching code:

| Key | Default | Description |
|-----|---------|-------------|
| `YOLO_MODEL` | `models/yolo11s.pt` | YOLO weights path |
| `DEPTH_CHECKPOINT` | `models/depth_pro.pt` | Depth Pro weights path |
| `DISTANCE_THRESHOLD` | `1.8` | Metres — closer than this = danger |
| `DANGER_CLASSES` | person, car, … | Which COCO classes to track |
| `DEPTH_INTERVAL` | `0.4` | Seconds between depth model runs |
| `ENABLE_BEEP` | `true` | Audio alert on/off |
| `SHOW_FPS` | `false` | FPS counter |
| `DEVICE` | `auto` | `cuda` or `cpu` |

---

## Platform Notes

| OS | Audio backend |
|----|--------------|
| Windows | `winsound.Beep` (built-in) |
| macOS | `afplay /System/Library/Sounds/Ping.aiff` |
| Linux | `paplay` → `aplay` → `beep` → terminal bell (fallback chain) |

---

## Performance Tips

- Use `DEVICE: cuda` for GPU acceleration (strongly recommended for Depth Pro).
- Increase `DEPTH_INTERVAL` (e.g., `0.8`) to run depth less often on slower hardware.
- Use a smaller YOLO model (`yolov8n.pt`) for faster detection.
- Reduce input resolution by resizing the frame before inference.

---

## License

MIT
