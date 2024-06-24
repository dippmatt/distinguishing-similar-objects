# Test YOLOv8 inference
from ultralytics import YOLO
from pathlib import Path

MODEL = 'yolov8s'  # yolov5n, yolov5s, yolov5m, yolov5l, yolov5x, custom 
EPOCHS = 30
IMAGE_REZ = 736
BATCH_SIZE = 24

model = YOLO(MODEL + '.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data=str('/dsimo/src/train/tless.yaml'), epochs=EPOCHS, imgsz=IMAGE_REZ, batch=BATCH_SIZE, project=str('/dsimo/runs/detect'))

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model('/home/matthias/Documents/distinguishing-similar-objects/dataset/t-less_v2/test_canon/01/rgb/0009.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')