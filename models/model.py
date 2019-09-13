import torch
import torch.nn.functional as F
from .model_blocks import GatedConv as g_conv
from .model_blocks import GatedDeconv as g_dconv
from .model_blocks import ResidualBlock as r_block
from .unet_parts import *

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def save(self, filename):
        state_dict = {name:value.cpu() for name, value \
                in self.state_dict().items()}
        status = {'state_dict':state_dict,}
        with open(filename, 'wb') as f_model:
            torch.save(status, f_model)

    def load(self, filename):
        if filename is None:
            raise ValueError('Error when loading model: filename not given.')
        status = torch.load(filename)
        self.load_state_dict(status['state_dict'])

class GatedConvModel(BaseModel):
    def __init__(self, in_channel):
        super(GatedConvModel, self).__init__()
        self._in_channel = in_channel
        self._build_network()
    def _build_network(self):
        c_num = 48
        self.gconv_layer_1 = torch.nn.Sequential(
                    g_conv(self._in_channel, c_num, 5, 1),
                    g_conv(c_num, 2*c_num, 3, 2),
                )
        self.gconv_layer_2 = torch.nn.Sequential(
                    g_conv(2*c_num, 2*c_num, 3, 1),
                    g_conv(2*c_num, 4*c_num, 3, 2),
                )
        self.gconv_layer_3 = torch.nn.Sequential(
                    g_conv(4*c_num, 4*c_num, 3, 1),
                    g_conv(4*c_num, 4*c_num, 3, 1),
                )
        self.gconv_layer_4 = torch.nn.Sequential(
                    g_conv(4*c_num, 4*c_num, 3, 1, dilation=2),
                    g_conv(4*c_num, 4*c_num, 3, 1, dilation=4),
                    g_conv(4*c_num, 4*c_num, 3, 1, dilation=8),
                    g_conv(4*c_num, 4*c_num, 3, 1, dilation=16),
                )
        self.gconv_layer_5 = torch.nn.Sequential(
                    g_conv(4*c_num, 4*c_num, 3, 1),
                    g_conv(4*c_num, 4*c_num, 3, 1),
                    g_dconv(4*c_num, 2*c_num, 3, 1),
                )
        self.gconv_layer_6 = torch.nn.Sequential(
                    g_conv(2*c_num, 2*c_num, 3, 1),
                    g_dconv(2*c_num, c_num, 3, 1),
                )
        self.gconv_layer_7 = torch.nn.Sequential(
                    g_conv(c_num, c_num, 3, 1),
                    g_conv(c_num, c_num//2, 3, 1),
                    g_conv(c_num//2, 1, 3, 1),
                )
        self.generator = torch.nn.Sequential(
                    self.gconv_layer_1,
                    self.gconv_layer_2,
                    self.gconv_layer_3,
                    self.gconv_layer_4,
                    self.gconv_layer_5,
                    self.gconv_layer_6,
                    self.gconv_layer_7,
                )

    def forward(self, x):
        out = self.generator(x)
        out = torch.clamp(out, min=0.)
        return out

class GatedConvSkipConnectionModel(GatedConvModel):
    def __init__(self, in_channel):
        super().__init__(in_channel)
        pass

    def forward(self, x):
        # skip connection : (1, 6), (2, 5) (3, 4)
        out1 = self.gconv_layer_1(x)
        out2 = self.gconv_layer_2(out1)
        out3 = self.gconv_layer_3(out2)
        out4 = self.gconv_layer_4(out3) + out2
        out5 = self.gconv_layer_5(out4) + out1
        out6 = self.gconv_layer_6(out5) 
        out = self.gconv_layer_7(out6)
        return out

class ResNet18(BaseModel):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self._build_network()
    def _build_network(self):
        # first layer

        c = 32
        self.pre = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channel, c, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU()
        )
        # residual layer * 4 (encoder)
        self.layer_1 = self._make_layer(c, c, 2)
        self.layer_2 = self._make_layer(c, c*2, 2)
        self.layer_3 = self._make_layer(c*2, c*4, 2)
        self.layer_4 = self._make_layer(c*4, c*8, 2)
        
        # residual layer * 4 (decoder)
        self.layer_5 = self._make_layer(c*8, c*4, 2)
        self.layer_6 = self._make_layer(c*4, c*2, 2)
        self.layer_7 = self._make_layer(c*2, c, 2)
        self.layer_8 = self._make_layer(c, 1, 2)

    def _make_layer(self, in_channel, out_channel, block_num):
        shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channel)
        )
        layers = []
        layers.append(r_block(in_channel, out_channel, kernel_size=3, padding=-1, shortcut=shortcut))
        for _ in range(1, block_num):
            layers.append(r_block(out_channel, out_channel, kernel_size=3, padding=-1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # first layer
        out = self.pre(x)

        # encoder
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        
        # decoder
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer_7(out)
        out = self.layer_8(out)

        return out

class ResNet18SkipConnection(ResNet18):
    def __init__(self, in_channel):
        super().__init__(in_channel)

    def forward(self, x):
        # first layer
        out = self.pre(x)

        # encoder
        out1 = self.layer_1(out)
        out2 = self.layer_2(out1)
        out3 = self.layer_3(out2)
        out4 = self.layer_4(out3)
        
        # decoder
        out5 = self.layer_5(out4) + out3
        out6 = self.layer_6(out5) + out2
        out7 = self.layer_7(out6) + out1
        out8 = self.layer_8(out7)

        return out8

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

