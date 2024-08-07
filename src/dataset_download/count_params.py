"""Preprocessing for Textureless RGB images by replcing RGB channels with different informations.

The first channel is used to store the original image in grayscale.
The second channel is the mask of the object, generated with segment anything (https://github.com/facebookresearch/segment-anything).

The third channel could either be the depth image or another preprocessing method, 
like small mask cutouts of the object after, e.g., applying harris corner detector.

"""
#from pycocotools import mask as mask_utils
#from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from ultralytics import FastSAM
from pathlib import Path
import matplotlib.pyplot as plt
import torch

device = "cuda"
model = FastSAM("FastSAM-s.pt")  # or FastSAM-s.pt


def _main():
    # state_dict = torch.load("/home/matthias/Downloads/FastSAM-s.pt")
    # model.load_state_dict(state_dict)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")


if __name__ == "__main__":
    _main()