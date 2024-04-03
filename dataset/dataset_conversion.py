import json
from pathlib import Path
import cv2 as cv

train_dir = Path(__file__).resolve().parent / Path("dataset/tless_train_pbr/tudl/train_pbr/").resolve()
save_path = Path("/home/matthias/Documents/datasets/t-less")
labels_save_path = save_path / Path("labels", "train2017")
images_save_path = save_path / Path("images", "train2017")

def _main():
    assert train_dir.exists(), f"{train_dir} does not exist" 
    # list all directories in the train_dir

    # In the training dataset, there are 50 scenes,
    # each scene is captured from 1000 different viewpoints
    scene_gt_info_file = Path("scene_gt_info.json")
    scene_gt_file = Path("scene_gt.json")

    scenes: list[Path] = []
    for obj_dir in train_dir.iterdir():
        if obj_dir.is_dir():
            scenes.append(Path(obj_dir))

    scenes.sort()
    # expect 50 scenes
    for scene in scenes:
        # get list of all rgb images in the scene (viewpoints)
        rgb_images = list(scene.glob("rgb/*.jpg"))
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

        # expect 1000 viewpoints here
        for key, viewpoint_obj_list in scene_gt_info.items():
            # get the rgb image for the viewpoint
            rgb_image_path = rgb_images[int(key)]
            assert rgb_image_path.exists(), f"{rgb_image_path} does not exist"
            rgb_image = cv.imread(str(rgb_images[int(key)]))

            # get the ground truth label ids for the viewpoint from scene_gt
            # get object ids as labels for the scene
            labels_list = list()
            for obj_info in scene_gt[key]:
                labels_list.append(obj_info["obj_id"])

            # get the bounding boxes for the objects in the viewpoint from scene_gt_info
            # expect the following keys in the list item: 
            # bbox_obj, bbox_visib, px_count_all, px_count_valid, px_count_visib, visib_fract
            # we expect 20 bboxes in the list, one for each object class.
            # if the class is not visible, the bbox is set to -1
            labels_list_str = list()
            for label_index, object in enumerate(viewpoint_obj_list):
                if object["bbox_visib"][0] == -1:
                    continue
                # Draw the bounding box on the image using OpenCV
                # Get the bounding box coordinates
                bbox_visib = object["bbox_visib"]
                x_min, y_min, delta_x, delta_y = bbox_visib[0], bbox_visib[1], bbox_visib[2], bbox_visib[3]
                x_max, y_max = x_min + delta_x, y_min + delta_y
                
                ##### For debugging purposes, draw the bounding box on the image using OpenCV #####
                # Draw the bounding box on the image using OpenCV
                # rgb_image = cv.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # draw the label name into the m_min, y_min coordinates
                # rgb_image = cv.putText(rgb_image, str(labels_list[label_index]), (x_min, y_min), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ###################################################################################

                # coco labels for the objects use [center_x, center_y, width, height] format normalized to [0, 1]
                # we need to convert the bounding box to this format
                # get the bounding box center
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                # normalize the bounding box coordinates
                # format to 6 decimal places
                center_x /= rgb_image.shape[1]
                center_x = round(center_x, 6)

                center_y /= rgb_image.shape[0]
                center_y = round(center_y, 6)

                width /= rgb_image.shape[1]
                width = round(width, 6)

                height /= rgb_image.shape[0]
                height = round(height, 6)

                # save the bounding box coordinates to a file
                # the file should be in the following format:
                # class_id center_x center_y width height
                labels_list_str.append(f"{labels_list[label_index]} {center_x} {center_y} {width} {height}\n")
                
            # save the image and label to the save_path
            viewpoint_string = rgb_images[int(key)].stem
            
            image_path = images_save_path / Path(f"{scene.name}_{viewpoint_string}.jpg")
            label_path = labels_save_path / Path(f"{scene.name}_{viewpoint_string}.txt")
            cv.imwrite(str(image_path), rgb_image)
            with open(label_path, "w") as f:
                f.writelines(labels_list_str)
            
            


        
            
      

if __name__ == "__main__":
    _main()