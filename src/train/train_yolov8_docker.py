# Test YOLOv8 inference
from ultralytics import YOLO
from pathlib import Path

PARAM_COUNT_5_CH = False
FIVE_CH_FROM_CHECKPOINT = True
FROM_CHECKPOINT = False
RESUME = False

MODEL = 'yolov8n'  # yolov8c, yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
EPOCHS = 30
IMAGE_REZ = 736
BATCH_SIZE = 32

# Initialize new model
#model = YOLO(MODEL +".yaml")
# Dont use pretrained weights, because they dont fit the modified model
if FIVE_CH_FROM_CHECKPOINT:
    model = YOLO('/dsimo/5k_backbone.pt')
elif FROM_CHECKPOINT:
    model = YOLO('/dsimo/runs/detect/train/weights/last.pt')
elif PARAM_COUNT_5_CH:
    model = YOLO(MODEL + '.yaml')
else:
    print("Using YoloV8s standard model")
    model = YOLO(MODEL + '.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
if RESUME:
    results = model.train(resume=True)
else:
    print("Training new model based on ultralytics backbone")
    results = model.train(data=str('/dsimo/src/train/tless.yaml'), epochs=EPOCHS, imgsz=IMAGE_REZ, batch=BATCH_SIZE, close_mosaic=10, project=str('/dsimo/runs/detect'))

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model('/home/matthias/Documents/distinguishing-similar-objects/dataset/t-less_v2/test_canon/01/rgb/0009.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')