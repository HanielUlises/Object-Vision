from ultralytics import YOLO

# model = YOLO("yolo11x.pt")
model = YOLO("yolo11n.pt")
# results = model.predict("input_videos/input_video.mp4", conf=0.2, save=True, show=True)
results = model.predict("input_videos/input_video.mp4", conf=0.2, save=True, presist=True)