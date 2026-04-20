import wandb
from ultralytics import YOLO

wandb.init(project="urbansense", name="yolo8n-nuscenes-baseline")

model = YOLO("yolov8n.pt")

results = model.train(
    data = "data/nuscenes_yolo.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    project="experiments",
    name="yolo_baseline",
)

wandb.log({"mAP50": results.results_dict["metrics/mAP50(B)"]})
wandb.finish()