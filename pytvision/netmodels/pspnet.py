import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.utils import model_zoo
from torchvision import models


__all__ = ["PSPNet", "pspnet"]


def pspnet(pretrained=False, **kwargs):
    r"""PSPNet model architecture"""
    model = PSPNet(**kwargs)
    if pretrained:
        pass
        # model.load_state_dict(model_zoo.load_url(model_urls['unet']))
    return model


class PSPDec(nn.Module):
    def __init__(self, in_features, out_features, downsize, upsize=60):
        super().__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(downsize, stride=downsize),
            nn.Conv2d(in_features, out_features, 1, bias=False),
            nn.BatchNorm2d(out_features, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(upsize),
        )

    def forward(self, x):
        return self.features(x)


class PSPNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=.95),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        """

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1
                m.requires_grad = False
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = False

        self.layer5a = PSPDec(2048, 512, 60)
        self.layer5b = PSPDec(2048, 512, 30)
        self.layer5c = PSPDec(2048, 512, 20)
        self.layer5d = PSPDec(2048, 512, 10)

        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, 1),
        )

    def forward(self, x):
        print("x", x.size())
        x = self.conv1(x)
        print("conv1", x.size())
        x = self.layer1(x)
        print("layer1", x.size())
        x = self.layer2(x)
        print("layer2", x.size())
        x = self.layer3(x)
        print("layer3", x.size())
        x = self.layer4(x)
        print("layer4", x.size())
        x = self.final(
            torch.cat(
                [
                    x,
                    self.layer5a(x),
                    self.layer5b(x),
                    self.layer5c(x),
                    self.layer5d(x),
                ],
                1,
            )
        )
        print("final", x.size())

        return F.upsample_bilinear(final, x.size()[2:])
