# Test YOLOv8 inference
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='t-less-reduced.yaml', epochs=25, imgsz=640, batch=32)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model('/home/matthias/Documents/distinguishing-similar-objects/dataset/t-less_v2/test_canon/01/rgb/0009.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')