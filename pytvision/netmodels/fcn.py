import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

class FCN8(nn.Module):

    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        feats = list(models.vgg16(pretrained=True).features.children())

        self.feats = nn.Sequential(*feats[0:9])
        self.feat3 = nn.Sequential(*feats[10:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False

        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat3 = self.feat3(feats)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat3 = self.score_feat3(feat3)
        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4
        score = F.upsample_bilinear(score, score_feat3.size()[2:])
        score += score_feat3

        return F.upsample_bilinear(score, x.size()[2:])

class FCN16(nn.Module):
    
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample_bilinear(score, x.size()[2:])

class FCN32(nn.Module):
    
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        self.feats = models.vgg16(pretrained=True).features
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        fconn = self.fconn(feats)
        score = self.score(fconn)

        return F.upsample_bilinear(score, x.size()[2:])

__all__ = ['FCN8', 'fcn8', 'FCN16', 'fcn16','FCN32', 'fcn32']

def fcn8(pretrained=False, **kwargs):
    r"""FCN8 model architecture
    """
    model = FCN8(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['unet']))
    return model

def fcn16(pretrained=False, **kwargs):
    r"""FCN16 model architecture
    """
    model = FCN16(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['unet']))
    return model

def fcn32(pretrained=False, **kwargs):
    r"""FCN32 model architecture
    """
    model = FCN32(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['unet']))
    return model

