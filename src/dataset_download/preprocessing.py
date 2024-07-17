"""Preprocessing for Textureless RGB images by replcing RGB channels with different informations.

The first channel is used to store the original image in grayscale.
The second channel is the mask of the object, generated with segment anything (https://github.com/facebookresearch/segment-anything).

The third channel could either be the depth image or another preprocessing method, 
like small mask cutouts of the object after, e.g., applying harris corner detector.

"""
from pycocotools import mask as mask_utils
#from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from pathlib import Path
import subprocess
import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

USE_FAST_SAM = True
device = "cuda"

# Largest model, 2.6 GB
model_type = "vit_h"
# Medium model, 1.2 GB
# model_type = "vit_l"
# Smallest model, 375 MB
# model_type = "vit_b"

# SAM

# Largest model, 2.6 GB
checkpoint = "/home/matthias/Downloads/sam_vit_h_4b8939.pth"
# Medium model, 1.2 GB
# checkpoint = "/home/matthias/Downloads/sam_vit_l_0b3195.pth"
# Smallest model, 375 MB
# checkpoint = "/home/matthias/Downloads/sam_vit_b_01ec64.pth"

# sam = sam_model_registry[model_type](checkpoint=checkpoint)
# sam.to(device=device)
# mask_generator = SamAutomaticMaskGenerator(sam)
# predictor = SamPredictor(sam)

# FastSAM

model = FastSAM("FastSAM-x.pt")  # or FastSAM-s.pt


# These are the directories the script exists to find in original_dataset_dir
# In each of them, a dataset in YOLO annotation format is expected (including images, labels subdirectories)
TRAIN_DATASET_NAME = "train_pbr_coco"
TEST_DATASET_NAME = "test_primesense_coco"

def get_fast_sam_anns(image_path: Path):
    anns = model(str(image_path), retina_masks=True, imgsz=736, conf=0.7, iou=0.8)
    annotation = anns[0]
    all_masks = list()
    for mask in annotation.masks:
        all_masks.append(mask.data.cpu().squeeze().numpy())
    # sort the masks by area
    all_masks = sorted(all_masks, key=(lambda x: x.sum()), reverse=True)
    # filter out any masks that are larger than 0.18 of the image area
    all_masks = [mask for mask in all_masks if mask.sum() < 0.10 * mask.shape[0] * mask.shape[1]]
    all_masks_np = np.stack(all_masks)

    return all_masks_np


def process_images(original_image_dir: Path, custom_image_dir: Path, set_name: str):
    i = 0
    sorted_images = sorted(list(original_image_dir.rglob("*.jpg")) + list(original_image_dir.rglob("*.png")))
    for image_path in tqdm.tqdm(sorted_images):
        # check if saved masks already exist
        if (custom_image_dir / f"{image_path.stem}_masks.npz").exists():
            print(f"Skipping {image_path.stem}")
            continue
        
        # open image as np array
        img = cv2.imread(str(image_path))
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if not USE_FAST_SAM:
            # SAM model [legacy]
            # Get segmentation option 1: use mask generator
            ann_list = mask_generator.generate(img)
            ann_list = sorted(ann_list, key=(lambda x: x['stability_score']), reverse=True)
            # limit to 40 best masks
            ann_list = ann_list[:40]
            # convertz all ann_list[i]['segmentation'] into one np array, the shape should be (number of masks, height, width)
            masks = np.stack([ann['segmentation'] for ann in ann_list])
            # save the masks as binary file, compress with np.savez_compressed
            np.savez_compressed(str(custom_image_dir / f"{image_path.stem}_masks.npz"), masks=masks)
        else:
            # get stacked array of shape (number of masks, height, width)
            masks = get_fast_sam_anns(image_path) # Fast SAM model            
            # convert to binary mask
            masks = masks > 0.5

            out_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
            mask_range = 128 // masks.shape[0]
            for i in range(masks.shape[0]):
                out_mask[masks[i]] = int((i * mask_range) + 128)
            
            image = Image.fromarray(out_mask)
            image.save(str(custom_image_dir / f"{image_path.stem}.png"))
            
        
        # load the masks
        #loaded_masks = np.load(str(custom_image_dir / f"{image_path.stem}_masks.npz"))
    
    return
    subprocess.run(["cp", str(image), str(custom_image_dir / image.name)])


def show_anns(anns):
    # filter anns and only show the ones with a certain iou
    print("len before", len(anns))
    anns = [ann for ann in anns if ann["stability_score"] > 0.98]
    print("len middle", len(anns))
    anns = [ann for ann in anns if ann["predicted_iou"] > 0.98]
    print("len after", len(anns))

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        # print(ann["stability_score"])
        # print(ann["stability_score"])
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        
        img[m] = color_mask
    ax.imshow(img)


def show_masks(masks):
    if len(masks) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks.shape[1], masks.shape[2], 4))
    img[:,:,3] = 0
    
    for ann in masks:
        m = ann
        
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        
        img[m] = color_mask
    ax.imshow(img)
    


def _main():

    # Notyfy the user that the script is gioing to overwrite any existing custom dataset
    # print("This script will overwrite any existing custom dataset.")
    # print("Enter YES to continue.")
    # user_input = input()
    # if user_input.lower() != "yes":
    #     print("Exiting.")
    #     import sys;sys.exit(0)

    original_dataset_dir = (Path(__file__).resolve().parent / Path("..", "..", "dataset", "t-less-original")).resolve()
    custom_dataset_dir = (Path(__file__).resolve().parent / Path("..", "..", "dataset", "t-less-custom")).resolve()

    assert original_dataset_dir.exists(), f"{original_dataset_dir} does not exist"
    assert custom_dataset_dir.exists(), f"{custom_dataset_dir} does not exist"
    
    training_images_in = original_dataset_dir / TRAIN_DATASET_NAME / "images"
    test_images_in = original_dataset_dir / TEST_DATASET_NAME / "images"

    training_images_out = custom_dataset_dir / TRAIN_DATASET_NAME / "images"
    test_images_out = custom_dataset_dir / TEST_DATASET_NAME / "images"

    training_labels_in = original_dataset_dir / TRAIN_DATASET_NAME / "labels"
    test_labels_in = original_dataset_dir / TEST_DATASET_NAME / "labels"

    training_labels_out = custom_dataset_dir / TRAIN_DATASET_NAME / "labels"
    test_labels_out = custom_dataset_dir / TEST_DATASET_NAME / "labels"

    # clean the output directories
    # subprocess.run(["rm", "-r", str(custom_dataset_dir / TRAIN_DATASET_NAME)])
    # subprocess.run(["rm", "-r", str(custom_dataset_dir / TEST_DATASET_NAME)])

    # recreate the directory structure
    training_images_out.mkdir(parents=True, exist_ok=True)
    test_images_out.mkdir(parents=True, exist_ok=True)

    training_labels_out.mkdir(parents=True, exist_ok=True)
    test_labels_out.mkdir(parents=True, exist_ok=True)

    assert training_images_in.exists(), f"{training_images_in} does not exist"
    assert test_images_in.exists(), f"{test_images_in} does not exist"
    assert training_labels_in.exists(), f"{training_labels_in} does not exist"
    assert test_labels_in.exists(), f"{test_labels_in} does not exist"
    assert training_images_out.exists(), f"{training_images_out} does not exist"
    assert test_images_out.exists(), f"{test_images_out} does not exist"
    assert training_labels_out.exists(), f"{training_labels_out} does not exist"
    assert test_labels_out.exists(), f"{test_labels_out} does not exist"


    process_images(training_images_in, training_images_out, "train")
    # process_images(test_images_in, test_images_out, "test")
    # copy labels
    subprocess.run(["cp", "-r", str(training_labels_in), str(training_labels_out)])
    subprocess.run(["cp", "-r", str(test_labels_in), str(test_labels_out)])

    print("Done")
    import sys;sys.exit(0)

    zipped_training_path = dataset_dir / Path("tless_train_pbr.zip")
    extracted_training_path = dataset_dir / Path("train_pbr")
    converted_training_path = dataset_dir / Path("train_pbr_coco")

    zipped_test_path = dataset_dir / Path("tless_test_primesense_bop19.zip")
    extracted_test_path = dataset_dir / Path("test_primesense")
    converted_test_path = dataset_dir / Path("test_primesense_coco")

    # find all jpg and png files in the converted_training_path
    training_images = list(converted_training_path.rglob("*.jpg")) + list(converted_training_path.rglob("*.png"))
    if len(training_images) > 0:
        print(Fore.GREEN + f"Found {len(training_images)} training images. Skipping download and conversion." + Fore.RESET)
    else:
        # Check if unzipped dataset exists
        if not extracted_training_path.exists():
            # Check if the zipped dataset exists
            if not zipped_training_path.exists():
                # Download the dataset
                download_dataset(bop_traing_url, zipped_training_path)
            # Unzip the dataset and convert it to COCO format
            unzip_dataset_and_convert(zipped_training_path, extracted_training_path)
        convert_bop_to_yolo(extracted_training_path, converted_training_path)


if __name__ == "__main__":
    _main()