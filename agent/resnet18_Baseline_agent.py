# torch
from torch import optim

# from my module
import depth_completion.models.model as model
from .depth_completion_agent import DepthCompletionAgent

class ResNet18SkipConnectionAgent(DepthCompletionAgent):
    def __init__(self, config):
        assert config['model_name'] == 'ResNet18SkipConnection'
        self.model = model.ResNet18SkipConnection(config['in_channel'])
        super().__init__(config, None)
