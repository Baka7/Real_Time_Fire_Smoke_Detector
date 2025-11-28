import cv2
import numpy as np
from ultralytics import YOLO

def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)

def draw_boxes(frame, results):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{r.names[cls]} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return frame
