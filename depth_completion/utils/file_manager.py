import os
import sys
import torch

class FileManager:
    def __init__(self):
        super(FileManager, self).__init__()

    def set_base_path(self, 
                      config, 
                      save_base_path=os.path.join(
                          os.path.dirname(sys.modules['__main__'].__file__), 
                          'experiments')):
        """Set base path when training mode"""
        self.save_filename = f"{config['dataset_name']}"\
                             f"_{config['model_name']}"\
                             f"_b{config['batch_size']}"\
                             f"_lr{config['lr']}"\
                             f"_{config['save_prefix']}"
        # save model path
        save_model_base_path = os.path.join(save_base_path, 'models')
        os.makedirs(save_model_base_path, exist_ok=True)
        # save log path
        logger_save_path = os.path.join(save_base_path, 'logs')
        os.makedirs(logger_save_path, exist_ok=True)
        # attribure
        self._log_path = os.path.join(logger_save_path, 
                                      self.save_filename+'.log')
        self.save_model_base_path = os.path.join(save_model_base_path, 
                                                 self.save_filename)

        os.makedirs(self.save_model_base_path, exist_ok=True)
        if os.path.exists(self._log_path):
            print(f'Warning: log file ({self._log_path}) has existed.')

    def save_log(self, msg):
        with open(self._log_path, 'a') as f:
            f.write(msg + '\n')

    def save_model(self, checkpoint):
        epoch = checkpoint['epoch']
        model_fname = os.path.join(self.save_model_base_path, f'epoch_{epoch}.pt')
        torch.save(checkpoint, model_fname)

    def load_model(self, path):
        return torch.load(path)

class MultiModelFileManager(FileManager):
    def __init__(self):
        """Overwrite FileManager __init__"""
        pass

    def set_base_path(self, 
                      config, 
                      save_base_path=os.path.join(
                          os.path.dirname(sys.modules['__main__'].__file__), 
                          'experiments')):
        """Set base path when training mode"""
        self.save_filename = f"{config['dataset_name']}_" + \
                             '_'.join([n for n in config['model_name']]) + \
                             f"_b{config['batch_size']}" + \
                             "_lr" + \
                             '_'.join([str(lr) for lr in config['lr']]) + \
                             f"_{config['save_prefix']}"
        # save model path
        save_model_base_path = os.path.join(save_base_path, 'models')
        os.makedirs(save_model_base_path, exist_ok=True)
        # save log path
        logger_save_path = os.path.join(save_base_path, 'logs')
        os.makedirs(logger_save_path, exist_ok=True)
        # attribure
        self._log_path = os.path.join(logger_save_path, 
                                      self.save_filename+'.log')
        self.save_model_base_path = os.path.join(save_model_base_path, 
                                                 self.save_filename)

        os.makedirs(self.save_model_base_path, exist_ok=True)
        if os.path.exists(self._log_path):
            print(f'Warning: log file ({self._log_path}) has existed.')
