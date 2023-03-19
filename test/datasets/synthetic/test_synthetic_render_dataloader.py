import os
import sys

import cv2
import numpy as np
import pytest

import torch
from pytvision import visualization as view

from pytvision.datasets.ellipse_dataset import SyntethicCircleDataset
from pytvision.transforms import transforms as mtrans
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Output parameters
PATHOUT = "/workspace/pytv/out"
FILENAME = "result_dataloader_{}.png"


def test_circle_dataloader():

    images_numbers = 300
    width, height = 512, 612
    sigma = 0.01
    batch_size = 3
    num_workers = 1

    data = SyntethicCircleDataset(
        count=images_numbers,
        generate=SyntethicCircleDataset.generate_image_mask_and_weight,
        imsize=(width, height),
        sigma=sigma,
        bdraw_grid=True,
        transform=transforms.Compose(
            [
                ## resize and crop
                mtrans.ToResize(
                    (400, 400),
                    resize_mode="square",
                    padding_mode=cv2.BORDER_REFLECT_101,
                ),
                # mtrans.CenterCrop( (200,200) ),
                # mtrans.RandomCrop( (255,255), limit=50, padding_mode=cv2.BORDER_REFLECT_101  ),
                # mtrans.ToResizeUNetFoV(388, cv2.BORDER_REFLECT_101),
                ## color
                mtrans.ToRandomChoiceTransform(
                    [
                        mtrans.RandomSaturation(),
                        mtrans.RandomHueSaturationShift(),
                        mtrans.RandomHueSaturation(),
                        # mtrans.RandomRGBShift(),
                        # mtrans.ToNegative(),
                        # mtrans.RandomRGBPermutation(),
                        # mtrans.ToRandomTransform( mtrans.ToGrayscale(), prob=0.5 ),
                        mtrans.ToGrayscale(),
                    ]
                ),
                ## blur
                # mtrans.ToRandomTransform( mtrans.ToLinealMotionBlur( lmax=1 ), prob=0.5 ),
                # mtrans.ToRandomTransform( mtrans.ToMotionBlur( ), prob=0.5 ),
                mtrans.ToRandomTransform(mtrans.ToGaussianBlur(), prob=0.75),
                ## geometrical
                # mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 )
                # mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 )
                mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REFLECT101),
                # mtrans.RandomGeometricalTransform( angle=360, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REFLECT101),
                # mtrans.RandomElasticDistort( size_grid=50, padding_mode=cv2.BORDER_REFLECT101 ),
                ## tensor
                mtrans.ToTensor(),
                mtrans.RandomElasticTensorDistort(size_grid=10, deform=0.05),
                ## normalization
                mtrans.ToNormalization(),
                # mtrans.ToWhiteNormalization(),
                # mtrans.ToMeanNormalization(
                #    mean=[0.485, 0.456, 0.406],
                #    std=[0.229, 0.224, 0.225]
                #    ),
            ]
        ),
    )

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    image_result_batched = []
    for i, x in enumerate(dataloader):
        print(
            i,
            x["image"].size(),
            x["label"].size(),
            x["weight"].size(),
        )

        # x_image \in R^[N,H,W,C]
        image = x["image"]
        label = x["label"]
        weight = x["weight"]

        print(torch.min(image), torch.max(image), image.shape)
        print(torch.min(label), torch.max(label), image.shape)
        print(torch.min(weight), torch.max(weight), image.shape)

        print(image.shape)
        print(np.unique(label))
        print(image.min(), image.max())

        image = image.permute(2, 3, 1, 0)[:, :, :, 0].squeeze().numpy()
        label = label.permute(2, 3, 1, 0)[:, :, :, 0].squeeze().numpy()
        weight = np.tile(weight.permute(2, 3, 1, 0)[:, :, :, 0], (1, 1, 3))

        # Create output folder
        PATHNAME = os.path.join(PATHOUT, FILENAME.format("circle_loader"))
        if os.path.exists(PATHOUT) is not True:
            os.makedirs(PATHOUT)

        # TODO February 12, 2023: change this for view function
        print(image.shape)
        print(label.shape)
        print(weight.shape)

        img_result = np.concatenate((image * 255, label * 255, weight * 10), axis=1)
        cv2.imwrite(PATHNAME, img_result)

        break


if __name__ == "__main__":
    pytest.main([__file__])
