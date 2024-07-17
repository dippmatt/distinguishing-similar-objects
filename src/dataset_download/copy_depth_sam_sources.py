from pathlib import Path
import subprocess
# import to print in color
from colorama import Fore
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

DEPTH = False

def img_depth_seg_2_out_image(img: Path, depth: Path, seg: Path):
    ref_shape = cv2.imread(str(img)).shape
    composed_image = np.zeros((ref_shape[0], ref_shape[1], 3), dtype=np.uint8)
    # now load each image and compose them
    img = cv2.imread(str(img))
    # convert to grayscale
    print(depth)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    depth = cv2.imread(str(depth), cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(str(seg), cv2.IMREAD_GRAYSCALE)

    # now compose each channel
    composed_image[:, :, 0] = img
    composed_image[:, :, 1] = depth
    composed_image[:, :, 2] = seg

    # show the composed image
    # plt.imshow(composed_image)
    # plt.show()
    # import sys;sys.exit(0)
    
    # now compose each 
    return composed_image

def npz_mask2png(npz_path: Path)-> np.ndarray:
    mask = np.load(str(npz_path))['masks']
    # the mask has the dimensions (n, 540, 720)
    # where n is the number of masks
    # we want to combine all masks into one image
    # first create a uint8 array with the same dimensions as the masks
    # then we evenly distribute the masks between 0 and 255

    out_mask = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)
    mask_range = 128 // mask.shape[0]
    for i in range(mask.shape[0]):
        out_mask[mask[i]] = int((i * mask_range) + 128)
    return out_mask

def main():
    # get the depth data from the random reduced dataset
    # dataset_dir = (Path(__file__).resolve().parent / Path("..", "..", "dataset", "t-less-original")).resolve()
    # reduced_training_path = dataset_dir / Path("train_pbr_reduced_coco")
    # original_tless_path = dataset_dir / Path("train_pbr")
    original_tless_path = Path("/media/matthias/Leer/dis-sim-obj-data/Dataset/BOP_DATASETS/tless/test_primesense")
    reduced_training_path = Path("/home/matthias/Documents/distinguishing-similar-objects/dataset/t-less-original/test_primesense_coco")
    
    assert reduced_training_path.exists(), f"Reduced training path {reduced_training_path} does not exist."
    #assert original_tless_path.exists(), f"Original T-LESS training path {original_tless_path} does not exist."

    # list all files in the reduced training path
    reduced_images = reduced_training_path / Path("images")
    reduced_training_images = list(reduced_images.rglob("*.jpg")) + list(reduced_images.rglob("*.png"))

    depth_output_path = reduced_training_path / Path("depth")
    depth_output_path.mkdir(parents=True, exist_ok=True)

    segment_src_path = Path("/home/matthias/Documents/distinguishing-similar-objects/dataset/t-less-custom/test_primesense_coco/images")
    segment_output_path = reduced_training_path / Path("segmentation")
    segment_output_path.mkdir(parents=True, exist_ok=True)

    composed_path = reduced_training_path / Path("composed")
    composed_path.mkdir(parents=True, exist_ok=True)


    # copy the depth images to the reduced training path
    for file in tqdm.tqdm(reduced_training_images):
        # print the name without type suffix
        bop_src_dir, image_name = file.stem.split("_")

        ############################## DEPTH IMAGE ##############################
        # # find the corresponding depth image in the original T-LESS dataset
        # depth_image_path = original_tless_path / Path(bop_src_dir) / Path("depth")
        # # find the image file regardless of the type suffix
        # found_images = list(depth_image_path.rglob(f"{image_name}.*"))
        # assert len(found_images) == 1, f"Found {len(found_images)} depth images for {image_name} in {depth_image_path}."
        # depth_image_path = found_images[0]

        # depth_image_out_path = depth_output_path / Path(f"{bop_src_dir}_{image_name}.png")
        # if not depth_image_out_path.exists():
        #     subprocess.run(["cp", str(depth_image_path), str(depth_image_out_path)])


        ############################## SEGMENTATION IMAGE ##############################
        # copy the segmentation mask from segment anything to the reduced training path
        # segmentation_imgage_src_path = segment_src_path / Path(f"{bop_src_dir}_{image_name}_masks.npz")
        # assert segmentation_imgage_src_path.exists(), f"Segmentation image {segmentation_imgage_src_path} does not exist."
        segmentation_image_out_path = segment_output_path / Path(f"{bop_src_dir}_{image_name}.png")
        #mask = npz_mask2png(segmentation_imgage_src_path)

        # save the mask as a png file using openCV
        # if not segmentation_image_out_path.exists():
        #     cv2.imwrite(str(segmentation_image_out_path), mask)

        depth_image_out_path = reduced_training_path / "corner_masks" / Path(f"{bop_src_dir}_{image_name}.png")
        # compose the depth image, the RGB image and the segmentation mask
        composed_image = img_depth_seg_2_out_image(file, depth_image_out_path, segmentation_image_out_path)
        composed_image_path = composed_path / Path(f"{bop_src_dir}_{image_name}.png")
        if not composed_image_path.exists():
            cv2.imwrite(str(composed_image_path), composed_image)
        

    
    # print finishing message
    print(Fore.GREEN + "Copied all depth images to the reduced training path." + Fore.RESET)
    return

if __name__ == '__main__':
    main()
