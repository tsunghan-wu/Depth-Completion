# builtin packages
import os
import numpy as np
import scipy.io as sio
from PIL import Image
from matplotlib import cm

# torch
import torch

# from my module
from depth_completion.utils import FileManager

class BaseAgent(object):
    def __init__(self, config=None, file_manager=None):
        """BaseAgent for training or testing

        Args:
            config: user provided, used for default `run` function.
        """
        self._config = config
        if file_manager is None:
            self._file_manager = FileManager()
        else:
            self._file_manager = file_manager

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def update(self, loss):
        raise NotImplementedError

    def save_model(self):
        checkpoint = {'epoch': self._epoch, 
                      'config': self._config,
                      'model_state_dict': {k: v.cpu() for k, v in \
                                           self.model.state_dict().items()},
                      'opt_state_dict': self.optimizer.state_dict()}
        self._file_manager.save_model(checkpoint)
    
    def load_model(self, load_model_path, param_only=False):
        checkpoint = self._file_manager.load_model(load_model_path)
        if isinstance(checkpoint['model_state_dict'], list):
            model = checkpoint['model_state_dict'][0]
            optimizer = checkpoint['opt_state_dict'][0]
        else:
            model = checkpoint['model_state_dict']
            optimizer = checkpoint['opt_state_dict']
        self.model.load_state_dict(model)
        if param_only == False:
            self._epoch = checkpoint['epoch']
            self._config = checkpoint['config']
            self.optimizer.load_state_dict(optimizer)

    def save_log(self, msg):
        self._file_manager.save_log(msg)

    def save_image(self, save_items, fnames):
        for key in save_items:  # go throungh different keys
            for tensor, fname in zip(save_items[key], fnames):    # batch items
                abs_fname = os.path.join(self._config['output'], f'{fname}_{key}.npy')
                img_np = tensor.squeeze().cpu().data.numpy()
                np.save(abs_fname, img_np)
