from pathlib import Path
import subprocess
# import to print in color
from colorama import Fore
import tqdm

def main():
    # get the depth data from the random reduced dataset
    dataset_dir = (Path(__file__).resolve().parent / Path("..", "..", "dataset", "t-less-original")).resolve()
    reduced_training_path = dataset_dir / Path("train_pbr_reduced_coco")
    original_tless_path = dataset_dir / Path("train_pbr")
    
    assert reduced_training_path.exists(), f"Reduced training path {reduced_training_path} does not exist."
    assert original_tless_path.exists(), f"Original T-LESS training path {original_tless_path} does not exist."

    # list all files in the reduced training path
    reduced_images = reduced_training_path / Path("images")
    reduced_training_images = list(reduced_images.rglob("*.jpg")) + list(reduced_images.rglob("*.png"))

    depth_output_path = reduced_training_path / Path("depth")
    depth_output_path.mkdir(parents=True, exist_ok=True)
    
    for file in tqdm.tqdm(reduced_training_images):
        # print the name without type suffix
        bop_src_dir, image_name = file.stem.split("_")

        # find the corresponding depth image in the original T-LESS dataset
        depth_image_path = original_tless_path / Path(bop_src_dir) / Path("depth")
        # find the image file regardless of the type suffix
        found_images = list(depth_image_path.rglob(f"{image_name}.*"))
        assert len(found_images) == 1, f"Found {len(found_images)} depth images for {image_name} in {depth_image_path}."
        depth_image_path = found_images[0]

        # rename the depth image to the same name as the RGB image
        depth_image_out_path = depth_output_path / Path(f"{bop_src_dir}_{image_name}.png")
        # copy the depth image to the reduced training path
        subprocess.run(["cp", str(depth_image_path), str(depth_image_out_path)])
    
    # print finishing message
    print(Fore.GREEN + "Copied all depth images to the reduced training path." + Fore.RESET)
    return

if __name__ == '__main__':
    main()
