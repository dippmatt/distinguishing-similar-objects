# Test YOLOv8 inference
from ultralytics import YOLO

MODEL = 'yolov8n'  # yolov5s, yolov5m, yolov5l, yolov5x, custom 

model = YOLO(MODEL + '.yaml')
# Load a pretrained YOLO model (recommended for training)
model = YOLO('/home/matthias/Documents/distinguishing-similar-objects/runs/detect/train/weights/best.pt')

# Use this to use validation data set (synthetic )
#results = model.val(split='val')
results = model.val(split='test')
# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.predict(source='/home/matthias/Downloads/t-less_v2_test_canon_01/01/rgb/0114.jpg', save_txt = True, save=True, project=".")