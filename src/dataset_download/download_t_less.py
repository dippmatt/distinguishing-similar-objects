import subprocess
from pathlib import Path
from colorama import Fore
from convert_bop2yolo_bbox import convert_bop_to_yolo
from create_dataset_subset import reduce_dataset    


def download_dataset(url: str, target_path: Path):
    dataset_pwd = target_path.parent
    if not target_path.exists():
        subprocess.run(["wget", "-P", str(dataset_pwd), url], cwd=dataset_pwd)
    else:
        print(Fore.GREEN + f"Skipping download of {target_path}.")
        print(Fore.RESET + f"{target_path} already exists.")

def unzip_dataset_and_convert(zipped_path: Path, target_path: Path):
    dataset_pwd = zipped_path.parent
    if not target_path.exists():
        subprocess.run(["unzip", str(zipped_path)], cwd=dataset_pwd)
    else:
        print(Fore.GREEN + f"Skipping conversion of {zipped_path} to COCO.")
        print(Fore.RESET + f"{target_path} already exists.")


# Get the BOP dataset, rendered scenes of T-LESS objects

def _main():
    dataset_dir = (Path(__file__).resolve().parent / Path("..", "..", "dataset", "t-less-original")).resolve()

    bop_traing_url = "https://bop.felk.cvut.cz/media/data/bop_datasets/tless_train_pbr.zip"
    bop_test_url = "https://bop.felk.cvut.cz/media/data/bop_datasets/tless_test_primesense_bop19.zip"

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

    # find all jpg and png files in the converted_test_path
    test_images = list(converted_test_path.rglob("*.jpg")) + list(converted_test_path.rglob("*.png"))
    if len(test_images) > 0:
        print(Fore.GREEN + f"Found {len(test_images)} test images. Skipping download and conversion." + Fore.RESET)
    else:
        # Check if unzipped dataset exists
        if not extracted_test_path.exists():
            # Check if the zipped dataset exists
            if not zipped_test_path.exists():
                # Download the dataset
                download_dataset(bop_test_url, zipped_test_path)
            # Unzip the dataset and convert it to COCO format
            unzip_dataset_and_convert(zipped_test_path, extracted_test_path)
        convert_bop_to_yolo(extracted_test_path, converted_test_path)

    # Create a reduced dataset for training, since the 50000 images can be too much
    reduced_training_path = dataset_dir / Path("train_pbr_reduced_coco")
    reduced_test_images = list(reduced_training_path.rglob("*.jpg")) + list(reduced_training_path.rglob("*.png"))
    if len(reduced_test_images) > 0:
        print(Fore.GREEN + f"Found {len(reduced_test_images)} as reduced training images. Skipping reduction." + Fore.RESET)
    else:
        if not reduced_training_path.exists():
            reduce_dataset(converted_training_path, reduced_training_path, 0.4, 42)

    return


if __name__ == "__main__":
    _main()