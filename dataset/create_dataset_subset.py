"""This script takes the T-LESS dataset and takes a subset of the 50k training images"""

import subprocess
from pathlib import Path
import random
from colorama import Fore
from tqdm import tqdm

def reduce_dataset(source_dataset: Path, target_dataset: Path, ratio: float = 0.3, seed: int = 42):
    assert source_dataset.exists(), f"{source_dataset} does not exist"
    dataset_name = source_dataset.stem
    labels_source_path = source_dataset / Path("labels")
    images_source_path = source_dataset / Path("images")
    assert labels_source_path.exists(), f"{labels_source_path} does not exist"
    assert images_source_path.exists(), f"{images_source_path} does not exist"

    if not target_dataset.exists():
        target_dataset.mkdir()
    labels_target_path = target_dataset / Path("labels")
    images_target_path = target_dataset / Path("images")
    if not labels_target_path.exists():
        labels_target_path.mkdir()
    if not images_target_path.exists():
        images_target_path.mkdir()

    # get the number of images in the source dataset
    source_images = list(images_source_path.glob("*.jpg")) + list(images_source_path.glob("*.png"))
    num_images = len(source_images)
    num_result_images = int(ratio * num_images)
    print(f"Number of images in the source dataset: {num_images}")
    print(Fore.GREEN + f"Reducing dataset {dataset_name} to {num_result_images}" + Fore.RESET)

    # now select num_result_images random indices within num_images
    target_indices = list(range(num_images))
    target_indices.sort()
    # now shuffle the indices to get a random selection, using a seed
    random.seed(seed)
    random.shuffle(target_indices)

    target_indices = target_indices[:num_result_images]
    target_indices.sort()
    
    for target_index in tqdm(target_indices):
        # copy the image
        source_image_path = source_images[target_index]
        source_label_path = str(source_image_path)
        source_label_path = source_label_path.replace(".jpg", ".txt").replace(".png", ".txt").replace("images", "labels")

        source_label_path = Path(source_label_path)
        assert source_label_path.exists(), f"{source_label_path} does not exist"
        assert source_image_path.exists(), f"{source_image_path} does not exist"

        # copy the label to labels_target_path
        subprocess.run(["cp", str(source_label_path), str(labels_target_path)], capture_output=True)

        # copy the image to images_target_path
        subprocess.run(["cp", str(source_image_path), str(images_target_path)])




if __name__ == "__main__":
    print("Skipping Script. Refer to Readme on how to get the dataset.")
    import sys; sys.exit(0)
    dataset_dir = Path(__file__).resolve().parent
    
    #import sys; sys.exit(0)
    source_dataset = dataset_dir / Path("test_primesense_coco")
    target_dataset = dataset_dir / Path("test_primesense_reduced_coco")
    reduce_dataset(source_dataset, target_dataset)