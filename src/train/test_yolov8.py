# Test YOLOv8 inference
from ultralytics import YOLO
from pathlib import Path
import numpy as np

eval_models = [
    #"/dsimo/runs/detect/ablation_sam_yolo8n/weights/best.pt",]
    "/dsimo/runs/detect/ablation_harris_yolo8n/weights/best.pt",]
#   "/dsimo/runs/detect/3-ch-default-baseline__2__-50k-yolo8n-v8.2.45/weights/best.pt"]
#    "/dsimo/runs/detect/ablation_gray_only_yolo8n/weights/best.pt", ]

# Initialize new model
# Dont use pretrained weights, because they dont fit the modified model
for model_path in eval_models:
    print("Model path", model_path)
    model_path = Path(model_path)
    model = YOLO(str(model_path))
    save_dir = model_path.parent.parent / "confusion_matrix.txt"

    # Evaluate the model's performance on the validation set
    results = model.val(save_json=True)

    confusion_matrix = results.confusion_matrix
    matrix_array = results.confusion_matrix.matrix
    np.savetxt(save_dir, matrix_array, delimiter=',')
    loaded_array = np.loadtxt(save_dir, delimiter=',')

    print("Loaded shape", loaded_array.shape)