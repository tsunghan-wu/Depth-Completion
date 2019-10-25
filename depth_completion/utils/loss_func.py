import torch
import torch.nn as nn
from depth_completion.utils import pytorch_ssim 

def depth_L2_loss(output, batch):
    output_depth = output['output_depth']
    render_depth = batch['render_depth']
    return nn.MSELoss()(output_depth, render_depth)

def depth_L1_loss(output, batch):
    output_depth = output['output_depth']
    render_depth = batch['render_depth']
    return nn.L1Loss()(output_depth, render_depth)

def bc_L2_loss(output, batch):
    output_boundary = output['output_boundary']
    depth_boundary = batch['depth_boundary']
    return nn.MSELoss()(output_boundary, depth_boundary)

def bc_L1_loss(output, batch):
    output_boundary = output['output_boundary']
    depth_boundary = batch['depth_boundary']
    return nn.L1Loss()(output_boundary, depth_boundary)


def depth_rel(output, batch):
    output_depth = output['output_depth']
    render_depth = batch['render_depth']
    diff = torch.abs((output_depth-render_depth))/torch.abs(render_depth)
    diff[torch.isnan(diff)] = 0 # remove nan
    loss = diff.median()
    return loss

def depth_ssim(output, batch):
    output_depth = output['output_depth']
    render_depth = batch['render_depth']
    ssim_loss = pytorch_ssim.SSIM(window_size = 11)
    return ssim_loss(output_depth, render_depth)

def miss(output, batch, threshold):
    output_depth = output['output_depth']
    render_depth = batch['render_depth']

    output_over_render, render_over_output \
        = output_depth/render_depth, render_depth/output_depth
    output_over_render[torch.isnan(output_over_render)] = 0
    render_over_output[torch.isnan(render_over_output)] = 0

    miss_map = torch.max(output_over_render, render_over_output)
    hit_rate = torch.sum(miss_map < threshold).float() / miss_map.numel()
    return hit_rate

def miss_105_1(output, batch):
    return miss(output, batch, 1.05)

def miss_110_1(output, batch):
    return miss(output, batch, 1.10)

def miss_125_1(output, batch):
    return miss(output, batch, 1.25)

def miss_125_2(output, batch):
    return miss(output, batch, 1.25**2)

def miss_125_3(output, batch):
    return miss(output, batch, 1.25**3)
