
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8s.pt')  # when executing this file will automatically download yolov8s.pt

# Train the model
model.train(
    data='data.yaml',  # Path to the YAML file
    epochs=50,          # Number of epochs to train
    batch=32,           # Batch size
    imgsz=640           # Image size for training
)
