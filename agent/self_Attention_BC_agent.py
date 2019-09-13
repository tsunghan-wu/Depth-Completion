import torch

from .mul_model_dc_agent import MultiModelDepthCompletionAgent as MMDCA


class BoundaryConsistencyUNetAgent(MMDCA):
    def __init__(self, config):
        super().__init__(config)

    def feed_into_net(self, batch, mode):
        # load into GPU
        if self.use_cuda: 
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device_ids[0])
        # process batch data
        color = batch['color']
        depth = batch['depth']
        normal = batch['normal']
        mask = batch['mask']
        boundary = batch['boundary']
        render_depth = batch['render_depth']

        # extract feature and feed into model
        feature = torch.cat([color, depth, normal, boundary, mask], dim=1)
        if self.use_cuda and mode == 'train':
            output_depth = torch.nn.parallel.data_parallel(
                self.models[0], feature, device_ids=self.device_ids)
            output_boundary = torch.nn.parallel.data_parallel(
                self.models[1], output_depth, device_ids=self.device_ids)
        else:
            output_depth = self.models[0](feature)
            output_boundary = self.models[1](output_depth)
        
        # process output
        output_mask = None
        render_depth_mask = torch.ones_like(render_depth)
        render_depth_mask[render_depth == 0] = 0
        ori_output_depth = output_depth
        output_depth = ori_output_depth * render_depth_mask
        output_boundary = output_boundary * render_depth_mask
        batch['depth_boundary'] = batch['depth_boundary'] * render_depth_mask
        output = {'output_depth': output_depth, 
                  'ori_output_depth': ori_output_depth,
                  'output_boundary': output_boundary}
        return output
