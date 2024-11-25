from ultralytics import YOLO

# Load a YOLO11n model
model = YOLO("runs/1/QYOLOv11/123/0.5RepViTm0_6+QLSKA11+QQiRMB(P5)(0.82)/weights/best.pt")

# Start tuning hyperparameters for YOLO11n training on the COCO8 dataset
result_grid = model.tune(data="1/data.yaml", use_ray=True, gpu_per_trial=1)