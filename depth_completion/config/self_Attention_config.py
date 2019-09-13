
common_config = {
    }

train_config = {
    "dataset_name": "matterport",
    "model_name": "GatedConvSkipConnectionModel",
    "in_channel": 9,
    "device_ids": [2],
    "seed": 7122,
    "num_workers": 8,

    "mode": "train",
    "train_path": "/tmp2/tsunghan/new_matterport/v1",
    "lr": 1e-4,
    "batch_size": 16,
    "loss_func": {('depth(L2)', 'depth_L2_loss', 1.), ('img_grad', 'img_grad_loss', 1e-3)},
    "load_model_path": None,
    "param_only": False,
    "validation": True,
    "valid_path": "/tmp2/tsunghan/new_matterport/v1",
    "epoches": 100,
    "save_prefix": "img_grad_first_try",
}


test_config = {
    "dataset_name": "matterport",
    "model_name": "GatedConvSkipConnectionModel",
    "in_channel": 9,
    "device_ids": [2],
    "seed": 7122,
    "num_workers": 8,

    "mode": "test",
    "test_path": "/tmp2/tsunghan/new_matterport/v1",
    "lr": 1e-4,
    "batch_size": 1,
    "loss_func": {('depth(L2)', 'depth_L2_loss', 1.), ('img_grad', 'img_grad_loss', 1e-3)},
    "load_model_path": "Depth-Completion/pre_train_model/self_Attention.pt",
    "param_only": True,
    "epoches": 100,
    "save_prefix": "img_grad_first_try",
    "output":"/tmp2/vis_dir_SA",
}
