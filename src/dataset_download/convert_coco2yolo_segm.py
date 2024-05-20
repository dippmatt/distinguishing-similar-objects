""" Convert the BOP dataset to YOLO format.

This script converts the BOP dataset to YOLO format. Ultralytics implementation (NEW APPROACH).
See convert_bop2yolo_bbox.py for a similar implementation using the custom scripting.
This implementation and creates the segmentation labels from a BOP TLESS dataset, 
once the COCO annotations were created using BOP Toolkit (see https://github.com/thodan/bop_toolkit/blob/master/scripts/calc_gt_coco.py).

"""

from ultralytics.data.converter import convert_coco
from pathlib import Path
import subprocess

# IMPORTANT: This script deletes some files in the input directory (tless_pbr_path). 
# Make sure to have a backup of the input directory. 

# Input directory
tless_pbr_path = Path('/home/matthias/Documents/distinguishing-similar-objects/dataset/yolo_conversion_in/test_primesense')
# Output directory
conversion_dir = Path('/home/matthias/Documents/distinguishing-similar-objects/dataset/yolo_conversion_out/test_primesense')

# create output directory structure
final_images_dir = conversion_dir / 'out' / 'images'
final_labels_dir = conversion_dir / 'out' / 'labels'
final_images_dir.mkdir(parents=True, exist_ok=True)
final_labels_dir.mkdir(parents=True, exist_ok=True)


# list all scene_gt_coco files in subdirectories of tless_pbr_path
scene_gt_coco_files = []
# get subdirectories
subdirs = [f for f in Path(tless_pbr_path).iterdir() if f.is_dir()]

subdirs.sort()
parent_dirs = [x.name for x in subdirs]
for subdir in subdirs:
    parent_dir = subdir.name

    # get the actual images
    images_dir = subdir / 'rgb'
    # list the images
    images = [f for f in images_dir.iterdir() if f.suffix.lower() == '.png' or f.suffix.lower() == '.jpg']
    # copy the images to the final images directory, but rename them to the parent directory name
    for image in images:
        image_name = image.name
        subprocess.run(['cp', image, final_images_dir / (parent_dir + '_' + image.name)])
        

    # get label definition file
    coco_gt_json = subdir / 'scene_gt_coco.json'
    # list all .json files in the subdirectory
    json_files = [f for f in subdir.iterdir() if f.suffix == '.json']
    # delete all json files except scene_gt_coco.json
    for json_file in json_files:
        if json_file != coco_gt_json:
            json_file.unlink()

    assert coco_gt_json.exists(), f"{coco_gt_json} does not exist"
    
    convert_coco(labels_dir=subdir, use_segments=True, save_dir=str(conversion_dir / Path("tmp", parent_dir)))
    # copy the labels to the final labels directory, but rename them to the parent directory name
    labels = [f for f in (conversion_dir / Path("tmp", parent_dir, 'labels', 'scene_gt_coco', 'rgb')).iterdir() if f.suffix == '.txt']
    for label in labels:
        assert label.exists(), f"{label} does not exist"
        subprocess.run(['cp', label, final_labels_dir / (parent_dir + '_' + label.name)])

