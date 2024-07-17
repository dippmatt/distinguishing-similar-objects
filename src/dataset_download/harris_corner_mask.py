import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm

VIEW = True

def corner_mask(image: Path, mask_radius=None):
    if mask_radius is None:
        mask_radius = np.sqrt(image.shape[0] * image.shape[1]) * 0.05
    img = cv2.imread(str(image))
    # use the Harris corner detector to find the corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur the image to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    dst = cv2.cornerHarris(gray_blur,2,3,0.04)

    threshold = 0.01 * dst.max()  # Adjust threshold as needed
    mask = np.zeros_like(gray_blur, np.uint8)
    mask[dst > threshold] = 255
    kernel = np.ones((mask_radius, mask_radius), np.uint8)
    mask = cv2.dilate(mask, kernel)
    # apply the mask on the gray image
    image = cv2.bitwise_and(gray, gray, mask=mask)

    if not VIEW:        
        return image
    else:
        print(mask.shape)
        print(mask.dtype)
        mask = mask / 255
        mask = mask.astype(bool)
        # overlay the mask on the image with a 0.2 opacity in red
        img[:,:,0][mask] = img[:,:,0][mask] * 0.5
        img[:,:,1][mask] = img[:,:,1][mask] * 0.5 
        
        cv2.imshow('dst',img)
        cv2.waitKey(0)

def main():
    dataset_dir = "/home/matthias/Documents/dsimo/dataset/t-less-original/train_pbr_coco/images"
    save_dir =    "/home/matthias/Documents/dsimo/dataset/t-less-custom/train_pbr_coco/corner_masks"
    # open first image in the dataset directory
    #list all images first
    images = list(Path(dataset_dir).rglob("*.jpg")) + list(Path(dataset_dir).rglob("*.png"))
    image_size = (540, 720)
    mask_radius = int(np.sqrt(image_size[0] * image_size[1]) * 0.05)
    i = 0
    for image in tqdm(images):
        image = Path("/home/matthias/Documents/dsimo/dataset/t-less-custom/test_primesense_coco/corner_masks/000013_000002.png")
        mask = corner_mask(image, mask_radius)
        import sys;sys.exit(0)
        # get the file name including the extension
        save_file = Path(save_dir) / image.name
        cv2.imwrite(str(save_file), mask)
        


    return

if __name__ == "__main__":
    main()