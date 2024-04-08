# Test YOLOv8 inference
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('/home/matthias/Documents/distinguishing-similar-objects/runs/detect/train2/weights/best.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.predict(source='/home/matthias/Downloads/t-less_v2_test_canon_01/01/rgb/0114.jpg', save_txt = True, save=True, project=".", name="test123")