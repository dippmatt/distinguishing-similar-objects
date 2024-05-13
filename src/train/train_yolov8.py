# Test YOLOv8 inference
from ultralytics import YOLO
from pathlib import Path

MODEL = 'yolov8n'  # yolov5n, yolov5s, yolov5m, yolov5l, yolov5x, custom 
REDUCED_DATASET = True

EPOCHS = 35

IMAGE_REZ = 736
BATCH_SIZE = 16

LOCAL_PRETRAINED = False
# ADAPT TO YOUR PATH
if LOCAL_PRETRAINED:
    BEST_WEIGHTS_PATH = "/home/matthias/Documents/distinguishing-similar-objects/runs/detect/train2/weights/best.pt"


if REDUCED_DATASET:
    config_name = "t-less-reduced"
    train_dir_name = "train_pbr_reduced_coco"
else:
    config_name = "t-less-full"
    train_dir_name = "train_pbr_coco"

val_dir_name = "test_primesense_coco"
test_dir_name = "test_primesense_coco"
    
# Output directory name for the training run
# UNUSED RIGHT NOW
# output_dir = Path(__file__).resolve().parent / Path("..", "..", "runs", config_name)
# if not output_dir.exists():
#     output_dir.mkdir(parents=True)

# Path to the dataset definition template
dataset_definition_template = Path(__file__).resolve().parent / Path("..", "..", "dataset", "t-less.yaml")
# Path to store the actual dataset definition after replacing the placeholder
dataset_definition_dir = (dataset_definition_template.parent / Path("gen_config")).resolve()
if not dataset_definition_dir.exists():
    dataset_definition_dir.mkdir(parents=True)
dataset_definition = dataset_definition_dir / Path(config_name + "_gen.yaml")

# path to dataset directory
dataset_directory = (Path(__file__).resolve().parent / Path("..", "..", "dataset", "t-less-original")).resolve()

train_dir = train_dir_name
val_dir = val_dir_name
test_dir = test_dir_name
# train_dir = Path(train_dir_name, "images")
# val_dir = Path(val_dir_name, "images")
# test_dir = Path(test_dir_name, "images")

assert dataset_definition.parent.exists(), f"{dataset_definition.parent} does not exist"
assert dataset_definition_template.exists(), f"{dataset_definition_template} does not exist"
assert dataset_directory.exists(), f"{dataset_directory} does not exist"
assert (dataset_directory / train_dir).resolve().exists(), f"{(dataset_directory / train_dir).resolve()} does not exist"
assert (dataset_directory / val_dir).resolve().exists(), f"{(dataset_directory / val_dir).resolve()} does not exist"

with open(dataset_definition_template, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "<DATASET_DIR_PLACEHOLDER>" in line:
        lines[i] = line.replace("<DATASET_DIR_PLACEHOLDER>", str(dataset_directory))
        continue
    if "<TRAIN_DIR_PLACEHOLDER>" in line:
        lines[i] = line.replace("<TRAIN_DIR_PLACEHOLDER>", str(train_dir))
        continue
    if "<VAL_DIR_PLACEHOLDER>" in line:
        lines[i] = line.replace("<VAL_DIR_PLACEHOLDER>", str(val_dir))
        continue
    if "<TEST_DIR_PLACEHOLDER>" in line:
        lines[i] = line.replace("<TEST_DIR_PLACEHOLDER>", str(test_dir))
        continue

with open(dataset_definition, "w") as f:
    f.writelines(lines)


# Create a new YOLO model from scratch
model = YOLO(MODEL + '.yaml')

# Load a pretrained YOLO model (recommended for training)
# use own pretrained model, adapt path!
if LOCAL_PRETRAINED:
    model = YOLO(BEST_WEIGHTS_PATH)
# use ultralytics pretrained model
else:
    model = YOLO(MODEL + '.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data=str(dataset_definition), epochs=EPOCHS, imgsz=IMAGE_REZ, batch=BATCH_SIZE) #, save_dir=str(output_dir))

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model('/home/matthias/Documents/distinguishing-similar-objects/dataset/t-less_v2/test_canon/01/rgb/0009.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')