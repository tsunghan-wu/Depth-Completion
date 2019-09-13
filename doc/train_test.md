# Training / Testing

Before training/testing, we hope that you have read [data](./data.md) and finish preparing dataset.

## Quick Start

For training/testing depth completion :

```bash
python3 -m depth_completion.main
```

Note : modify `main.py` for different agent and configuration if needed.

## Agent

For different agent, you can see [agent\_list](../depth_completion/agent/__init__.py) and then modify `main.py`.

## Configuration

For different configuration corresponding to the agent, you can see [configs](../depth_completion/config/) for example configurations. Here we will take one for example :

```
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
```

### Common

- model\_name : Models in [models](../depth_completion/models/model.py) you use for backbone model. (Second input for boundary consistency network)
- in\_channel : input channel for the model
- device\_ids : GPU-ids (support multi-gpu training)

### Training

- training\_path, valid\_path : training and validation dataset root path. (Training list and Testing list are [here](../depth_completion/data/data_list))
- loss\_func : dictionary for loss functions, each element is a tuple : (description, function\_name, weight). You can see [loss\_funcs](../depth_completion/utils/loss_func.py) for all loss functions.
- param\_only : Set False as default. If set as True, you need to pass load\_model\_path for re-training.

#### Note
- Training log and model will be saved in [experiments](../depth_completion/experiments) directory.

### Testing

- test\_path : testing dataset path.
- batch\_size : set as 1 for testing
- param\_only : Set as True and pass `load\_model\_path`.
- output : output directory saving `.npy` files.

#### Note 
- You can download pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1gmQS2mkIs9KO4m-eTI1zfTnXiqNQYbTp?usp=sharing) if needed.
- Testing scripts only give you `.npy` files. If you want to evaluate the result or visualize them, please refer [visualize](./visualize.md)
