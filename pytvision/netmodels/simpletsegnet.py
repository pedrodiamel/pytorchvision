

import torch
import torch.nn as nn


__all__ = ['SimpleSegNet', 'simpletsegnet']

def simpletsegnet(pretrained=False, **kwargs):
    r"""SimpleSegNet model architecture
    """
    model = SimpleSegNet(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['unet']))
    return model



class SimpleSegNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=1, filters_base=32, patch_border=16):
        super(SimpleSegNet, self).__init__()
        s = filters_base

        self.patch_border = patch_border
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.input_conv = BasicConv2d(num_channels, s, 1)
        self.enc_1 = BasicConv2d(s * 1, s * 2, 3, padding=1)
        self.enc_2 = BasicConv2d(s * 2, s * 4, 3, padding=1)
        self.enc_3 = BasicConv2d(s * 4, s * 8, 3, padding=1)
        self.enc_4 = BasicConv2d(s * 8, s * 8, 3, padding=1)
        # https://github.com/pradyu1993/segnet - decoder lacks relu (???)
        self.dec_4 = BasicConv2d(s * 8, s * 8, 3, padding=1)
        self.dec_3 = BasicConv2d(s * 8, s * 4, 3, padding=1)
        self.dec_2 = BasicConv2d(s * 4, s * 2, 3, padding=1)
        self.dec_1 = BasicConv2d(s * 2, s * 1, 3, padding=1)
        self.conv_final = nn.Conv2d(s, num_classes, 1)

    def forward(self, x):
        # Input
        x = self.input_conv(x)
        # Encoder
        x = self.enc_1(x)
        x = self.pool(x)
        x = self.enc_2(x)
        x = self.pool(x)
        x = self.enc_3(x)
        x = self.pool(x)
        x = self.enc_4(x)
        # Decoder
        x = self.dec_4(x)
        x = self.upsample(x)
        x = self.dec_3(x)
        x = self.upsample(x)
        x = self.dec_2(x)
        x = self.upsample(x)
        x = self.dec_1(x)
        # Output
        x = self.conv_final(x)
        
        #b = self.patch_border
        #x = F.sigmoid(x[:, :, b:-b, b:-b])

        return x 



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

