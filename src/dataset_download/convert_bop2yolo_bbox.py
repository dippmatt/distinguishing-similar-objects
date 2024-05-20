""" Convert the BOP dataset to YOLO format.

This script converts the BOP dataset to YOLO format. Custom implementation (old apporach).
See convert_coco2yolo_segm.py for a similar implementation using the ultralytics library ().
This is a custom scripting implementation and creates the bounding boxes 
as labels directly from a BOP TLESS dataset.

"""


import json
from pathlib import Path
import subprocess
#import cv2 as cv
from tqdm import tqdm
from colorama import Fore

def convert_bop_to_yolo(unzipped_dataset: Path, target_path: Path):
    assert unzipped_dataset.exists(), f"{unzipped_dataset} does not exist"
    
    # Create the directories for the YOLO dataset
    if not target_path.exists():
        target_path.mkdir()
    assert target_path.exists(), f"{target_path} does not exist"
    images_save_path = target_path / Path("images")
    labels_save_path = target_path / Path("labels")
    if not images_save_path.exists():
        images_save_path.mkdir()
    if not labels_save_path.exists():
        labels_save_path.mkdir()

    # Get the Ground Truth Information for the dataset
    scene_gt_info_file = Path("scene_gt_info.json")
    scene_gt_file = Path("scene_gt.json")

    # each scene is a collection of viewpoints of the same scenario
    scenes: list[Path] = []
    for obj_dir in unzipped_dataset.iterdir():
        if obj_dir.is_dir():
            scenes.append(Path(obj_dir))

    scenes.sort()
    print(Fore.GREEN + f"Converting BOP dataset {unzipped_dataset.stem} to YOLO format." + Fore.RESET)
    ###########################
    # labels_set = set()
    ###########################    
    for scene in tqdm(scenes):
        # get list of all rgb images in the scene (viewpoints)
        rgb_images = list(scene.glob("rgb/*.jpg")) + list(scene.glob("rgb/*.png"))
        rgb_images.sort()

        # json data containing the ground truth information for the scene
        scene_gt = None
        with open(scene / scene_gt_info_file, "r") as f:
            scene_gt_info = json.load(f)
        if not scene_gt_info:
            raise FileNotFoundError(f"{scene / scene_gt_info_file} not found")
        with open(scene / scene_gt_file, "r") as f:
            scene_gt = json.load(f)
        if not scene_gt:
            raise FileNotFoundError(f"{scene / scene_gt_file} not found")

        # each viewpoint is a rgb image with ground truth information
        # print(scene_gt_info.keys())
        # import sys;sys.exit(0)
        i = 0
        for key, viewpoint_obj_list in scene_gt_info.items():
            # get the rgb image
            rgb_image_path = rgb_images[i]
            image_dtype = rgb_image_path.suffix
            i += 1
            assert rgb_image_path.exists(), f"{rgb_image_path} does not exist"

            # get the image shape
            rgb_image_metadata = subprocess.run(["identify", str(rgb_image_path)], capture_output=True)
            assert rgb_image_metadata.returncode == 0, f"Error: {rgb_image_metadata.stderr}"
            rgb_image_shape = rgb_image_metadata.stdout.decode("utf-8").split()[2].split("x")
            img_width, img_height = int(rgb_image_shape[0]), int(rgb_image_shape[1])

            # get the ground truth label ids for the viewpoint from scene_gt
            # get object ids as labels for the scene
            labels_list = list()
            for obj_info in scene_gt[key]:
                labels_list.append(obj_info["obj_id"] - 1)

            # get the bounding boxes for the objects in the viewpoint from scene_gt_info
            # expect the following keys in the list item: 
            # bbox_obj, bbox_visib, px_count_all, px_count_valid, px_count_visib, visib_fract
            # we expect 20 bboxes in the list, one for each object class.
            # if the class is not visible, the bbox is set to -1
            labels_list_str = list()

            # Load the image using OpenCV for debugging the bounding box drawing
            # rgb_image = cv.imread(str(rgb_image_path))

            ##########################
            # labels_set = set(labels_list).union(labels_set)
            # continue
        
            ##########################
            for label_index, object in enumerate(viewpoint_obj_list):
                if object["bbox_visib"][0] == -1:
                    continue
                # Draw the bounding box on the image using OpenCV
                # Get the bounding box coordinates
                bbox_visib = object["bbox_visib"]
                x_min, y_min, delta_x, delta_y = bbox_visib[0], bbox_visib[1], bbox_visib[2], bbox_visib[3]
                x_max, y_max = x_min + delta_x, y_min + delta_y

                # Draw the bbox on the image using OpenCV for debugging purposes
                # rgb_image = cv.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # rgb_image = cv.putText(rgb_image, str(labels_list[label_index]), (x_min, y_min), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # yolo labels for the objects use [center_x, center_y, width, height] format normalized to [0, 1]
                # we need to convert the bounding box to this format
                # get the bounding box center
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                # normalize the bounding box coordinates
                # format to 6 decimal places
                center_x /= img_width
                center_x = round(center_x, 6)

                center_y /= img_height
                center_y = round(center_y, 6)

                bbox_width /= img_width
                bbox_width = round(bbox_width, 6)

                bbox_height /= img_height
                bbox_height = round(bbox_height, 6)

                # save the bounding box coordinates to a file
                # the file should be in the following format:
                # class_id center_x center_y width height
                labels_list_str.append(f"{labels_list[label_index]} {center_x} {center_y} {bbox_width} {bbox_height}\n")
            
            # Save the image with the bounding boxes for debugging
            # cv.imwrite("test.jpg", rgb_image)
            # print("Image saved to test.jpg")
            # import sys;sys.exit(0)

            # save the image and label to the save_path
            viewpoint_string = rgb_image_path.stem
            
            image_path = images_save_path / Path(f"{scene.name}_{viewpoint_string}{image_dtype}")
            label_path = labels_save_path / Path(f"{scene.name}_{viewpoint_string}.txt")
            #cv.imwrite(str(image_path), rgb_image)
            # instead of cv.imwrite, we just copy the image
            subprocess.run(["cp", str(rgb_image_path), str(image_path)])
            with open(label_path, "w") as f:
                f.writelines(labels_list_str)
    #########################
    # print(f"Labels set: {labels_set}")
    # return
    #########################
    
    print(f"Images saved to {target_path}\n")
    return


def _main():
    print("Main is disabled. Refer to Readme on how to get the dataset.")
    return
    unzipped_dataset = Path(__file__).resolve().parent / Path("test_primesense")
    assert unzipped_dataset.exists(), f"{unzipped_dataset} does not exist"
    target_path = Path(__file__).resolve().parent / Path("test_primesense_coco")

    if not target_path.exists():
        target_path.mkdir()
    assert target_path.exists(), f"{target_path} does not exist"
        
    convert_bop_to_coco(unzipped_dataset, target_path)
    return

if __name__ == "__main__":
    _main()