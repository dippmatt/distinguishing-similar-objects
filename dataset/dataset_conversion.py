import subprocess
import yaml
from pathlib import Path


train_dir = Path(__file__).resolve().parent / Path("t-less_v2/train_canon/").resolve()

def _main():
    assert train_dir.exists(), f"{train_dir} does not exist" 
    # list all directories in the train_dir
        
    for obj_dir in train_dir.iterdir():
        if obj_dir.is_dir():
            # get the ground truth file
            gt_path = obj_dir / "gt.yml"
            assert gt_path.exists(), f"{gt_path} does not exist"
            with open(gt_path, "r") as f:
                gt = yaml.load(f, Loader=yaml.CLoader)
                print(gt)
            
      

if __name__ == "__main__":
    _main()