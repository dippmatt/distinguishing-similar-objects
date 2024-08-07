FROM ultralytics/ultralytics:latest

# Install additional dependencies
RUN apt-get update && apt-get install -y nano imagemagick
RUN python3 -m pip install tqdm colorama line-profiler
# RUN python3 -m pip uninstall -y albumentations
