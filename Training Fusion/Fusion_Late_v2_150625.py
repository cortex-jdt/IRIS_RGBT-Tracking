import os
import torch
import numpy as np
import cv2
from torchvision.ops import nms
from ultralytics import YOLO

# ===== CONFIGURATION =====

# Load models
model_rgb = YOLO('/home/javier/Javier/runs/detect/train_yolo11n_LLVIP-FLIR-Tokyo_Visible_1_080625/weights/best.pt')
model_thermal = YOLO('/home/javier/Javier/runs/detect/train_yolo11n_LLVIP-FLIR-Tokyo_Infrared_1_070625/weights/best.pt')

# Define folders
rgb_folder = '/media/act/Javier_datasets/CVC14/CVC-14/Night/Visible/NewTest/FramesPos'
thermal_folder = '/media/act/Javier_datasets/CVC14/CVC-14/Night/FIR/NewTest/FramesPos'
output_folder = '/home/javier/Javier/Fusion_Late_v2_Result'
fusion_label_folder = '/home/javier/Javier/Fusion_Late_v2_Labels'

# Create output folders if not exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(fusion_label_folder, exist_ok=True)

# Process only .tif files
rgb_filenames = sorted([f for f in os.listdir(rgb_folder) if f.lower().endswith('.tif')])

# ===== FUSION LOOP =====

for filename in rgb_filenames:
    rgb_path = os.path.join(rgb_folder, filename)
    thermal_path = os.path.join(thermal_folder, filename)

    # Load RGB image
    frame_rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if frame_rgb is None:
        print(f"⚠ Skipping: Cannot read RGB tif: {rgb_path}")
        continue

    # Normalize if 16-bit
    if frame_rgb.dtype == np.uint16:
        frame_rgb = cv2.normalize(frame_rgb, None, 0, 255, cv2.NORM_MINMAX)
        frame_rgb = np.uint8(frame_rgb)

    # Convert RGB to 3-channel if not already
    if len(frame_rgb.shape) == 2 or frame_rgb.shape[2] == 1:
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)

    # Load Thermal image
    frame_thermal = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
    if frame_thermal is None:
        print(f"⚠ Skipping: Cannot read Thermal tif: {thermal_path}")
        continue

    if frame_thermal.dtype == np.uint16:
        frame_thermal = cv2.normalize(frame_thermal, None, 0, 255, cv2.NORM_MINMAX)
        frame_thermal = np.uint8(frame_thermal)

    # Convert Thermal to 3-channel
    if len(frame_thermal.shape) == 2 or frame_thermal.shape[2] == 1:
        frame_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_GRAY2RGB)

    # Resize thermal to match RGB size
    frame_thermal = cv2.resize(frame_thermal, (frame_rgb.shape[1], frame_rgb.shape[0]))

    # ===== INFERENCE =====

    results_rgb = model_rgb(frame_rgb)
    results_thermal = model_thermal(frame_thermal)

    # Extract detections from RGB
    boxes_rgb = results_rgb[0].boxes.xyxy.cpu().numpy()
    scores_rgb = results_rgb[0].boxes.conf.cpu().numpy()
    classes_rgb = results_rgb[0].boxes.cls.cpu().numpy()
    detections_rgb = np.hstack([boxes_rgb, scores_rgb[:, np.newaxis], classes_rgb[:, np.newaxis]])

    # Extract detections from Thermal
    boxes_thermal = results_thermal[0].boxes.xyxy.cpu().numpy()
    scores_thermal = results_thermal[0].boxes.conf.cpu().numpy()
    classes_thermal = results_thermal[0].boxes.cls.cpu().numpy()
    detections_thermal = np.hstack([boxes_thermal, scores_thermal[:, np.newaxis], classes_thermal[:, np.newaxis]])

    # Apply optional weighting
    detections_rgb[:, 4] *= 0.5
    detections_thermal[:, 4] *= 0.5

    # Merge detections and apply NMS
    all_detections = np.vstack([detections_rgb, detections_thermal])
    boxes = torch.tensor(all_detections[:, :4])
    scores = torch.tensor(all_detections[:, 4])
    keep_indices = nms(boxes, scores, iou_threshold=0.5)
    final_detections = all_detections[keep_indices.numpy()]

    # Get image size for YOLO label normalization
    img_height, img_width = frame_rgb.shape[:2]

    # Save YOLO-format fusion predictions
    label_filename = os.path.splitext(filename)[0] + '.txt'
    label_path = os.path.join(fusion_label_folder, label_filename)

    with open(label_path, 'w') as f:
        for det in final_detections:
            x1, y1, x2, y2, conf, cls = det
            cx = (x1 + x2) / 2 / img_width
            cy = (y1 + y2) / 2 / img_height
            w = (x2 - x1) / img_width
            h = (y2 - y1) / img_height
            f.write(f"{int(cls)} {cx} {cy} {w} {h}\n")

    # Visualization on RGB image
    for det in final_detections:
        x1, y1, x2, y2, conf, cls = det
        label = f'{int(cls)} {conf:.2f}'
        cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame_rgb, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save visualization result as .png
    output_filename = os.path.splitext(filename)[0] + '.png'
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, frame_rgb)

    print(f"✅ Processed {filename}")

print("🎯 Fusion completed for all available image pairs.")