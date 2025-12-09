import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import depth_pro
import urllib.request

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_dir = "checkpoints"
checkpoint_path = os.path.join(checkpoint_dir, "depth_pro.pt")
checkpoint_url = "https://huggingface.co/apple/DepthPro/resolve/main/depth_pro.pt"

if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 100_000_000:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Downloading DepthPro checkpoint...")
    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
    print("Checkpoint downloaded successfully.")

image_path = "Images/image1.jpeg"

yolo_model = YOLO("yolo11s.pt")
image_input = cv2.imread(image_path)
results = yolo_model(image_input)

person_boxes = []
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    for box, cls in zip(boxes, classes):
        if result.names[int(cls)] == "person":
            x1, y1, x2, y2 = map(int, box[:4])
            person_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image_input, (x1, y1), (x2, y2), (0, 255, 0), 2)

model, transform = depth_pro.create_model_and_transforms()
model.to(device)
model.eval()

rgb_image, _, f_px = depth_pro.load_rgb(image_path)
rgb_image = transform(rgb_image).to(device)

with torch.no_grad():
    prediction = model.infer(rgb_image, f_px=f_px)

depth_np = prediction["depth"].squeeze().cpu().numpy()

for x1, y1, x2, y2 in person_boxes:
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    depth_value = float(depth_np[center_y, center_x])
    text = f"Depth: {depth_value:.2f} m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = x1, y1 - 10
    cv2.rectangle(
        image_input,
        (text_x - 5, text_y - text_size[1] - 10),
        (text_x + text_size[0] + 5, text_y + 5),
        (0, 0, 0),
        -1
    )
    cv2.putText(image_input, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

cv2.imshow("Detections + Depth", image_input)
cv2.imwrite("output_depth.jpg", image_input)
cv2.waitKey(0)

depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
depth_color = cv2.applyColorMap((255 - depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

cv2.imshow("Depth Map", depth_color)
cv2.imwrite("output_depth_colormap.jpg", depth_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
