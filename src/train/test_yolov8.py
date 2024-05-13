# Test YOLOv8 inference
from ultralytics import YOLO

MODEL = 'yolov8n'  # yolov5s, yolov5m, yolov5l, yolov5x, custom 

# Load a pretrained YOLO model (recommended for training)
model = YOLO('/home/matthias/Documents/distinguishing-similar-objects/runs/detect/train4/weights/last.pt')

# Use this to use validation data set (synthetic )
#results = model.val(split='val')
results = model.val(split='test')
