from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from pathlib import Path
import random
from PIL import Image
import numpy as np 

# Create a FastSAM model
model = FastSAM("FastSAM-x.pt")  # or FastSAM-x.pt

image = "/dsimo/dataset/t-less-custom/test_primesense_coco/images/000018_000399.png" 

source = str(image)
# Run inference on an image
anns = model(source, retina_masks=True, imgsz=736, conf=0.7, iou=0.8)
# print("Len masks:", len(everything_results[0].masks))
# import sys;sys.exit(0)
# prompt_process = FastSAMPrompt(source, everything_results, device=0)
# # Everything prompt
# anns = prompt_process.everything_prompt()
print("LEN:", len(anns))
for ann in anns:
    print("Len masks:", len(ann.masks))
    print("Class names:", ann.names)
    print("Keypoints:", ann.keypoints)
    print("Probs:", ann.probs)
    print("Probs:", ann.probs)
    ann.save("masks.png")
    for mask in ann.masks:
        print("Mask type:", type(mask))
        print("Mask shape:", mask.shape)
        print(mask)
        # save mask.data to file using matplotlib
        image = mask.data
        np_array = image.cpu().squeeze().numpy()
        np_array = (np_array - np_array.min()) / (np_array.max() - np_array.min()) * 255
        np_array = np_array.astype(np.uint8)
        image = Image.fromarray(np_array)
        image.save('output_image.png')
        import sys;sys.exit(0)

# prompt_process.plot(annotations=ann, output="./")

