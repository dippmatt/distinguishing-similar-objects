import torch
import onnxruntime as ort
model = "/home/matthias/Documents/DIS_SIM_OBJ_DOKU/runs/3-ch-default-baseline-50k-yolo8n-v8.2.45/weights/last.pt"
ort_session = ort.InferenceSession("your_model.onnx")



# model = torch.load("/home/matthias/Documents/DIS_SIM_OBJ_DOKU/runs/3-ch-default-baseline-50k-yolo8n-v8.2.45/weights/last.pt")
# total_params = sum(p.numel() for p in model.parameters())
# print("Total Parameters:", total_params)
# print("Total Parameters:", total_params)
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Trainable Parameters:", trainable_params)