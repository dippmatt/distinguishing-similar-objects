SUBMODULE_URL = https://github.com/ultralytics/ultralytics
SUBMODULE_PATH = ./ultralytics
SUBMODULE_TAG = v8.2.0

init:	
	git submodule update --init --recursive
	cd $(SUBMODULE_PATH) && git checkout tags/$(SUBMODULE_TAG)