import subprocess
from pathlib import Path

dataset_dir = Path(__file__).resolve().parent

# Get the BOP dataset, rendered scenes of T-LESS objects
#subprocess.run(["wget", "-P", "https://bop.felk.cvut.cz/media/data/bop_datasets/tless_train_pbr.zip"], cwd=dataset_dir)
subprocess.run(["wget", "-P", str(dataset_dir), "https://bop.felk.cvut.cz/media/data/bop_datasets/tless_models.zip"], cwd=dataset_dir)
# wait for the download to finish
subprocess.run(["unzip", "tless_models.zip"], cwd=dataset_dir)
#subprocess.run(["unzip", "tless_train_pbr.zip"], cwd=dataset_dir)
