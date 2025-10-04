from ultralytics import YOLO
model = YOLO("yolo11n.yaml").load("yolo11n.pt")