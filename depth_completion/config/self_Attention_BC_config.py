# Boundary Consistency Unet config

common_config = {
    }

train_config = {
    "dataset_name": "matterport",
    "model_name": ("GatedConvSkipConnectionModel", "UNet"),
    "in_channel": (9, 1),
    "device_ids": [3],
    "seed": 7122,

    "num_workers": 8,
    "mode": "train",
    "train_path": "/tmp2/tsunghan/new_matterport/v1",
    "lr": (1e-4, 1e-4),
    "batch_size": 2,
    "loss_func": {('depth(L1)', 'depth_L1_loss', 1.), ('bc(L1)', 'bc_L1_loss', 1)},
    "load_model_path": (None, None),
    "param_only": (False, False),
    "validation": True,
    "valid_path": "/tmp2/tsunghan/new_matterport/v1",
    "epoches": 100,
    "save_prefix": "official_ver",
}

test_config = {
    "dataset_name": "matterport",
    "model_name": ("GatedConvSkipConnectionModel", "UNet"),
    "in_channel": (9, 1),
    "device_ids": [0],
    "seed": 7122,

    "num_workers": 8,
    "mode": "test",
    "test_path": "/work/kaikai4n/new_matterport/v1",
    "lr": (1e-4, 1e-4),
    "batch_size": 4,
    "loss_func": {('depth(L2)', 'depth_L2_loss', 1.), ('bc(L2)', 'bc_L2_loss', 1)},
    "load_model_path": (None, None),
    "param_only": (True, True),
    "save_prefix": "test",
    "output":"vis_dir",
}
