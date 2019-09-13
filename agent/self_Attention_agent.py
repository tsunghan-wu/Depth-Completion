# torch
from torch import optim

# from my module
import depth_completion.models.model as model
from .depth_completion_agent import DepthCompletionAgent

class GatedConvSkipConnectionAgent(DepthCompletionAgent):
    def __init__(self, config):
        assert config['model_name'] == 'GatedConvSkipConnectionModel'
        self.model = model.GatedConvSkipConnectionModel(config['in_channel'])
        super().__init__(config, None)
