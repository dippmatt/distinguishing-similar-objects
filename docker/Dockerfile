FROM ultralytics/ultralytics:8.2.26

# Install additional dependencies
RUN apt-get update && apt-get upgrade
RUN apt-get install -y imagemagick
RUN python3 -m pip install tqdm colorama
