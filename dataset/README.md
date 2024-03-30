# Get the rendered T-LESS Training data scene 
This one includes several objects rendered in synthetic scenes

[BOP Challenge 2020 T-LESS](https://bop.felk.cvut.cz/datasets/#T-LESS)

# Get the original T-LESS dataset 
This one only includes individual objects for training

The script to download the T-Less dataset just run `python3 t-less_download.py` from this directory.
Beware, the entire dataset is 63 GB in size.

[T-LESS Dataset](https://cmp.felk.cvut.cz/t-less/)

# T-Less toolkit

The [T-LESS Toolkit](https://github.com/thodan/t-less_toolkit/tree/master) is supposed to automate training scene generation.
However, the environment is buggy with python3.10 and there are no python package reqirements specified.
I therefore removed the toolkit submodule from this repo, since it's a hassle to use.
Solution: use BOP Challenge training data instead.