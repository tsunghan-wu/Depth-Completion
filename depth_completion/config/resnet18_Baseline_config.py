
common_config = {
    }

train_config = {
    "dataset_name": "matterport",
    "model_name": "ResNet18SkipConnection",
    "in_channel": 9,
    "device_ids": [0],
    "seed": 7122,

    "num_workers": 8,
    "mode": "train",
    "train_path": "/tmp2/tsunghan/new_matterport/v1",
    "lr": 1e-4,
    "batch_size": 8,
    "loss_func": {('depth(L2)', 'depth_L2_loss', 1.)},
    "load_model_path": None,
    "param_only": False,
    "validation": True,
    "valid_path": "/tmp2/tsunghan/new_matterport/v1",
    "epoches": 100,
    "save_prefix": "",
}

test_config = {
    "dataset_name": "matterport",
    "model_name": "ResNet18SkipConnection",
    "in_channel": 9,
    "device_ids": [0, 1, 2, 3],
    "seed": 7122,
    "num_workers": 8,

    "mode": "test",
    "test_path": "/tmp2/tsunghan/new_matterport/v1",
    "lr": 1e-4,
    "batch_size": 1,
    "loss_func": {('depth(L2)', 'depth_L2_loss', 1.), ('img_grad', 'img_grad_loss', 1e-3)},
    "load_model_path": "/tmp2/tsunghan/twcc_data/twcc_experience_resnet/matterport_ResNet18SkipConnection_b10_lr0.0001_/epoch_13.pt",
    "param_only": True,
    "epoches": 100,
    "save_prefix": "resnet",
    "output":"/tmp2/tsunghan/experiment_result/mat_npy/r18sc_epo13",
}
