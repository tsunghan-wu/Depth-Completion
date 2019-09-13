import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                groups=1, bias=False, shortcut=None):
        super().__init__()
        if padding == -1:
            padding = ((np.array(kernel_size) - 1) * np.array(dilation)) // 2
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
        )
        self.right = shortcut
    
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        if padding == -1:
            padding = tuple(((np.array(kernel_size) - 1) * np.array(dilation)) // 2)
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class NN3Dby2D(object):
    '''
    Use these inner classes to mimic 3D operation by using 2D operation frame by frame.
    '''
    class Base(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, xs):
            dim_length = len(xs.shape)
            if dim_length == 5:  # [batch_size, channels, video_len, w, h]
                # Unbind the video data to a tuple of frames
                xs = torch.unbind(xs, dim=2)
                # Process them frame by frame using 2d layer
                xs = torch.stack([self.layer(x) for x in xs], dim=2)
            elif dim_length == 4:  # [batch_size, channels, w, h]
                # keep the 2D ability when the data is not batched videoes but batched frames
                xs = self.layer(xs)
            return xs

    class Conv3d(Base):
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride, padding, dilation, groups, bias):
            super().__init__()
            # take off the kernel/stride/padding/dilation setting for the temporal axis
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[1:]
            if isinstance(stride, tuple):
                stride = stride[1:]
            if isinstance(padding, tuple):
                padding = padding[1:]
            if isinstance(dilation, tuple):
                dilation = dilation[1:]
            self.layer = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias
            )

            # let the spectral norm function get its conv weights
            self.weight = self.layer.weight
            # let partial convolution get its conv bias
            self.bias = self.layer.bias

    class BatchNorm3d(Base):
        def __init__(self, out_channels):
            super().__init__()
            self.layer = nn.BatchNorm2d(out_channels)

    class InstanceNorm3d(Base):
        def __init__(self, out_channels, track_running_stats=True):
            super().__init__()
            self.layer = nn.InstanceNorm2d(out_channels, track_running_stats=track_running_stats)


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=-1, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True), conv_by='2d'):
        super().__init__()
        if conv_by == '2d':
            module = NN3Dby2D
        elif conv_by == '3d':
            module = torch.nn
        else:
            raise NotImplementedError(f'conv_by {conv_by} is not implemented.')

        if padding == -1:
            padding = ((np.array(kernel_size) - 1) * np.array(dilation)) // 2
        self.gatingConv = module.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.featureConv = module.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = module.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = module.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.gatingConv = nn.utils.spectral_norm(self.gatingConv)
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, xs):
        gating = self.gatingConv(xs)
        feature = self.featureConv(xs)
        if self.activation:
            feature = self.activation(feature)
        out = self.gated(gating) * feature
        if self.norm is not None:
            out = self.norm_layer(out)
        return out

class GatedDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=-1, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 scale_factor=2, conv_by='2d'):
        super().__init__()
        self.conv = GatedConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by=conv_by)
        self.scale_factor = scale_factor

    def forward(self, xs):
        xs_resized = F.interpolate(xs, scale_factor=(self.scale_factor, self.scale_factor))
        return self.conv(xs_resized)


