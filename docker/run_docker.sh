#!/bin/bash

# print working directory
SCRIPTDIR="$( dirname -- "$BASH_SOURCE[0]" )"

# Docker run options explained:

# -it: interactive
# --ipc=host: access to all of vram & ram
# -u $MY_UID: run container as current shell user instead of root
# --rm: remove container after use
# --gpus all: make gpus available
# --volume="$SCRIPTDIR/..:/dsimo:rw": make this repository visible under /dsimo
# --volume="$SCRIPTDIR/startup.sh:/startup.sh:ro": mount initialisation script for container startup
# --entrypoint /startup.sh: run initialisation script for container startup
# container name
#-u $USER \

docker run \
-it \
--ipc=host \
--gpus all \
--rm \
--volume="$SCRIPTDIR/..:/dsimo:rw" \
--volume="$SCRIPTDIR/startup.sh:/startup.sh:ro" \
--entrypoint /startup.sh \
--name=dsimo_v2_container dsimo_v2

