"""This script takes the T-LESS dataset and takes a subset of the 50k training images"""

import subprocess
from pathlib import Path
import cv2 as cv
from PIL import Image
import random

# set the maximum number of images to be used for training
TRAIN_SIZE_LIMIT_REL = 0.3


# path to the full training dataset
source_dataset = Path("/home/matthias/Documents/datasets/t-less")
labels_source_path = source_dataset / Path("labels", "train2017")
images_source_path = source_dataset / Path("images", "train2017")
assert labels_source_path.exists(), f"{labels_source_path} does not exist"
assert images_source_path.exists(), f"{images_source_path} does not exist"

target_dataset = Path("/home/matthias/Documents/datasets/t-less-reduced")
labels_target_path = target_dataset / Path("labels", "train2017")
images_target_path = target_dataset / Path("images", "train2017")
assert labels_target_path.exists(), f"{labels_target_path} does not exist"
assert images_target_path.exists(), f"{images_target_path} does not exist"


def _main():
    assert source_dataset.exists(), f"{source_dataset} does not exist" 
    # get the number of images in the source dataset
    source_images = list(images_source_path.glob("*.jpg"))
    num_images = len(source_images)
    num_result_images = int(TRAIN_SIZE_LIMIT_REL * num_images)
    print(f"Number of images in the source dataset: {num_images}")
    print(f"Reducing the dataset to {num_result_images}")

    # now select num_result_images random indices within num_images
    target_indices = list(range(num_images))
    target_indices.sort()
    # now shuffle the indices to get a random selection, using a seed
    random.seed(42)
    random.shuffle(target_indices)

    target_indices = target_indices[:num_result_images]
    target_indices.sort()
    
    for target_index in target_indices:
        # copy the image
        source_image_path = source_images[target_index]
        source_label_path = str(source_image_path).replace(".jpg", ".txt").replace("images", "labels")
        source_label_path = Path(source_label_path)
        assert source_label_path.exists(), f"{source_label_path} does not exist"
        assert source_image_path.exists(), f"{source_image_path} does not exist"

        # copy the label to labels_target_path
        subprocess.run(["cp", str(source_label_path), str(labels_target_path)])
        # copy the image to images_target_path
        subprocess.run(["cp", str(source_image_path), str(images_target_path)])
        
        
        

      

if __name__ == "__main__":
    _main()