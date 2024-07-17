import torch

# Define paths to the source and target model .pt files
source_model_path = '/dsimo/runs/detect/20k_3_ch_original_dataset/weights/best.pt'
target_model_path = '/dsimo/runs/detect/20k_5_ch_custom_dataset_1_epoch/weights/best.pt'
updated_target_model_path = '/dsimo/5k_backbone.pt'

# Load the source & target model
source_model = torch.load(source_model_path)
target_model = torch.load(target_model_path)

# Extract the state_dict (weights) from the source model
source_state_dict = source_model['ema'].state_dict()
target_state_dict = target_model['ema'].state_dict()

# print the layer shapes of the source model

source_keys = list(source_state_dict.keys())
target_keys = list(target_state_dict.keys())

for i, key in enumerate(source_keys):
    source_shape = source_state_dict[source_keys[i]].shape
    target_shape = target_state_dict[target_keys[i]].shape
    if int(source_keys[i].split(".")[1]) != 0:
        assert source_shape == target_shape, f"Source shape {source_shape} and target shape {target_shape} do not match"
    # copy the weights from the source model to the target model
    else:
        target_state_dict[target_keys[i]] = source_state_dict[source_keys[i]]

torch.save(target_model, updated_target_model_path)
