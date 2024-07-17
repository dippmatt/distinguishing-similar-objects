from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from pathlib import Path
import random

# Create a FastSAM model
model = FastSAM("FastSAM-x.pt")  # or FastSAM-x.pt

# create different validation results for 10 confident thresholds and 10 iou thresholds
iou_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

src_dir = "/dsimo/dataset/t-less-custom-reduced-2k-images/test_primesense_coco/images" 
# Define 100 random sample images in an inference source directory
images = list(Path(src_dir).rglob("*.png"))
# select 100 random images
images_sampled = random.sample(list(images), 100)

output_base_dir = Path("./fast_sam_calibration")

for iou in iou_thresholds:
    for conf in conf_thresholds:
        for image in images_sampled:
            source = str(image)
            # Run inference on an image
            everything_results = model(source, retina_masks=True, imgsz=736, conf=conf, iou=iou)
            prompt_process = FastSAMPrompt(source, everything_results, device=0)
            # Everything prompt
            ann = prompt_process.everything_prompt()
            output_dir = output_base_dir / f"iou_{iou}_conf_{conf}"
            output_dir.mkdir(parents=True, exist_ok=True)
            prompt_process.plot(annotations=ann, output=str(output_dir))
