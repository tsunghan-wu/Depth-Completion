import torch
import torch.nn as nn
from .torch_utils import img_grad, sobel_conv
from depth_completion.utils import pytorch_ssim 

def smooth_depth_loss(output, batch):
    output_depth = output['output_depth']
    gpu_id = output_depth.get_device()
    loss = torch.tensor(0.).to(gpu_id)
    _, _, w, h = output_depth.shape
    _dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]   # four adjacent pixels
    for i in range(1, w-1):
        for j in range(1, h-1):
            for k in _dir:
                dx, dy = k
                tmp_loss = nn.MSELoss()(output_depth[:, 0, i+dx, j+dy] - output_depth[:, 0, i, j], torch.zeros_like(output_depth[:, 0, i, j]))
                loss += tmp_loss
    return loss

def depth_L2_loss(output, batch):
    output_depth = output['output_depth']
    render_depth = batch['render_depth']
    return nn.MSELoss()(output_depth, render_depth)

def depth_L1_loss(output, batch):
    output_depth = output['output_depth']
    render_depth = batch['render_depth']
    return nn.L1Loss()(output_depth, render_depth)

def img_grad_loss(output, batch):
    output_depth, boundary = output['output_depth'], batch['depth_boundary']
    depth_grad = sobel_conv(output_depth) 
    loss = torch.mean(depth_grad * boundary)
    return -loss

def bc_L2_loss(output, batch):
    output_boundary = output['output_boundary']
    depth_boundary = batch['depth_boundary']
    return nn.MSELoss()(output_boundary, depth_boundary)

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
