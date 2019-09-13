from abc import abstractmethod
from torch import optim

import depth_completion.models.model as model
from depth_completion.utils import MultiModelFileManager
from .depth_completion_agent import DepthCompletionAgent

class MultiModelDepthCompletionAgent(DepthCompletionAgent):
    def __init__(self, config, file_manager=None):
        if file_manager is None:
            file_manager = MultiModelFileManager()
        super(DepthCompletionAgent, self).__init__(config, file_manager)
        self.models = []
        for model_name, ch in zip(config['model_name'], config['in_channel']):
            model_obj = getattr(model, model_name)
            self.models.append(model_obj(ch))

        self.use_cuda = True if len(self._config['device_ids']) >= 1 else False
        self.device_ids = self._config['device_ids']
        self.optimizers = []
        for i in range(len(self.models)):
            if self.use_cuda is True:
                self.models[i] = self.models[i].to(self.device_ids[0])
            self.optimizers.append(optim.Adam(
                self.models[i].parameters(), lr=self._config['lr'][i]))
        self._epoch = 0
        for i, option in enumerate(self._config['load_model_path']):
            if option is not None:
                param_only = self._config['param_only'][i]
                self.load_model(option, param_only, i)
                if param_only == False:
                    new_optimizer = optim.Adam(self.models[i].parameters(), 
                                                lr=self._config['lr'][i])  
                    new_optimizer.load_state_dict(
                        self.optimizers[i].state_dict())
                    self.optimizer = new_optimizer
        print(self._epoch)
        print (self._config)
        
    @abstractmethod
    def feed_into_net(self, x, mode):
        pass
    
    def load_model(self, load_model_path, param_only, i):
        checkpoint = self._file_manager.load_model(load_model_path)
        self.models[i].load_state_dict(checkpoint['model_state_dict'][i])
        if param_only == False:
            self._epoch = checkpoint['epoch']
            self._config = checkpoint['config']
            self.optimizers[i].load_state_dict(checkpoint['opt_state_dict'][i])

    def save_model(self):
        model_num = len(self.models)
        checkpoint = {'epoch': self._epoch, 
                      'config': self._config,
                      'model_state_dict': [{k: v.cpu() for k, v in \
                          self.models[i].state_dict().items()}\
                          for i in range(model_num)],
                      'opt_state_dict': [self.optimizers[i].state_dict()\
                          for i in range(model_num)]}
        self._file_manager.save_model(checkpoint)

    def change_model_state(self, state):
        if state == 'train':
            for i in range(len(self.models)):
                self.models[i].train()
        elif state == 'eval':
            for i in range(len(self.models)):
                self.models[i].eval()

    def update(self, loss):
        for i in range(len(self.optimizers)):
            self.optimizers[i].zero_grad()
        loss.backward()
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

