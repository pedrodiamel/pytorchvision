import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# ----------------------------------------------------------------------------------------------
# FMP
# Image based Static Facial Expression Recognition with Multiple Deep Network Learning
# https://www.andrew.cmu.edu/user/yzhiding/publications/ICMI15.pdf
# https://www.microsoft.com/en-us/research/publication/image-based-static-facial-expression-recognition-with-multiple-deep-network-learning/
# https://azure.microsoft.com/en-us/services/cognitive-services/emotion/
# Pedro D. Marrero Fernandez
# ----------------------------------------------------------------------------------------------

__all__ = ["FMPNet", "fmp"]


def fmp(pretrained=False, **kwargs):
    r"""FMP model architecture
    https://www.andrew.cmu.edu/user/yzhiding/publications/ICMI15.pdf
    """
    model = FMPNet(**kwargs)
    if pretrained:
        pass
        # model.load_state_dict(model_zoo.load_url(model_urls['fmp']))
    return model


class FMPNet(nn.Module):
    def __init__(self, num_classes=8, num_channels=1, init_weights=True, batch_norm=False):
        super(FMPNet, self).__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.size_input = 48
        self.dim = 128 * 6 * 6

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(0.5, 0.5)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(0.5, 0.5)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(0.5, 0.5)),
            nn.BatchNorm2d(128),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def representation(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
