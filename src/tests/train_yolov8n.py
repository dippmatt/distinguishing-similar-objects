# Test YOLOv8 inference
from ultralytics import YOLO
from pathlib import Path

# Path to the dataset definition template
dataset_definition_template = Path(__file__).resolve().parent / Path("..", "..", "dataset", "t-less-reduced.yaml")
assert dataset_definition_template.exists(), f"{dataset_definition_template} does not exist"
with open(dataset_definition_template, "r") as f:
    lines = f.readlines()

# Path to store the actual dataset definition after replacing the placeholder
dataset_definition = dataset_definition_template.parent / Path("t-less-reduced_gen.yaml")
# path to dataset directory
dataset_directory = (Path(__file__).resolve().parent / Path("..", "..", "dataset")).resolve()
assert dataset_directory.exists(), f"{dataset_directory} does not exist"

for i, line in enumerate(lines):
    if "<DATASET_DIR_PLACEHOLDER>" in line:
        lines[i] = line.replace("<DATASET_DIR_PLACEHOLDER>", str(dataset_directory))
        break

with open(dataset_definition, "w") as f:
    f.writelines(lines)

# Create a new YOLO model from scratch
model = YOLO('yolov8s.yaml')

# Load a pretrained YOLO model (recommended for training)
#model = YOLO('yolov8n.pt')
model = YOLO('yolov8s.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data=str(dataset_definition), epochs=50, imgsz=640, batch=32)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model('/home/matthias/Documents/distinguishing-similar-objects/dataset/t-less_v2/test_canon/01/rgb/0009.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')