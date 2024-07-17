#!/bin/bash

echo "DSIMO STARTUP MESSAGE"
cd /dsimo
rm -rf /usr/src/ultralytics/* && cp -r ultralytics/* /usr/src/ultralytics/ && echo 'Copied custom ultralytics code.'
#cp -r /usr/src/ultralytics/* ultralytics_8.2.45/ && echo 'Copied custom ultralytics code.'
bash

