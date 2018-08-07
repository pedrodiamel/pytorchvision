import torch
import torch.nn as nn
import torch.nn.init as init
import math


__all__ = ['UNet', 'unet']

def unet(pretrained=False, **kwargs):
    r"""UNet model architecture
    """
    model = UNet(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['unet']))
    return model


class UNet(nn.Module):

    def __init__(self, num_classes=1, in_channels=3, is_deconv=False, is_batchnorm=False):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        down1,befdown1 = self.down1(x)
        down2,befdown2 = self.down2(down1)
        down3,befdown3 = self.down3(down2)
        down4,befdown4 = self.down4(down3)       
        center = self.center(down4)
        up4 = self.up4(befdown4, center)
        up3 = self.up3(befdown3, up4)
        up2 = self.up2(befdown2, up3)
        up1 = self.up1(befdown1, up2)
        y = self.final(up1)

        return y


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 0),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 0),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs1 = self.down(outputs)
        return outputs1,outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, 2)
        else:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear')

    def forward(self, inputs1, inputs2):
        ch_select=inputs2.size()[1] // 2
        up_in=inputs2[:,0:ch_select,:,:]
        outputs2 = self.up(up_in)
        offset = inputs1.size()[2] - outputs2.size()[2]
        padding = [offset // 2, offset // 2 ]
        outputs1 = inputs1[:,:,padding[0]:-padding[0],padding[1]:-padding[1]]
        return self.conv(torch.cat([outputs1, outputs2], 1))