# Test pytorch version and check if CUDA is available

import torch

# print PyTorch version
print(f"PyTorch version: {torch.__version__}")
# check if CUDA is available
print(f"Is CUDA available?: {torch.cuda.is_available()}")
# check VRAM
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3} GB")

