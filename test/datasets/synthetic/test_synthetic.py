import os

import cv2
import numpy as np
import pytest

from pytvision.transforms.ellipserender import CircleRender, EllipseRender

# Output parameters
PATHOUT = "/workspace/pytv/out"
FILENAME = "result_render_{}.png"


def test_ellipse_render():
    """
    Test ellipse render
    """
    # Render parameters
    width = 512
    height = 512
    cnt = 5
    rmin = 5
    rmax = 50
    border = 90
    sigma = 0.2

    # Execute render
    img, label, meta = EllipseRender.generate(width, height, cnt, rmin, rmax, border, sigma, True)

    h, w, _ = img.shape
    assert h == height
    assert w == width

    print(label.shape)
    print(meta)

    # Create output folder
    PATHNAME = os.path.join(PATHOUT, FILENAME.format("ellipse"))
    if os.path.exists(PATHOUT) is not True:
        os.makedirs(PATHOUT)

    # Save result
    label_transform = np.tile(label.sum(axis=0, keepdims=True).transpose((1, 2, 0)), (1, 1, 3))
    print(label_transform.shape)

    img_result = np.concatenate(
        (
            img[:, :, (2, 1, 0)],
            label_transform * 255,
        ),
        axis=1,
    )
    cv2.imwrite(PATHNAME, img_result)

    print("DONE!!!")


def test_circle_render():
    """
    Test circle render
    """
    # Render parameters
    width = 512
    height = 512
    cnt = 5
    rmin = 5
    rmax = 50
    border = 90
    sigma = 0.2

    img, label, meta = CircleRender.generate(width, height, cnt, rmin, rmax, border, sigma, True)

    h, w, _ = img.shape
    assert h == height
    assert w == width

    print(label.shape)
    print(meta)

    # Create output folder
    PATHNAME = os.path.join(PATHOUT, FILENAME.format("circle"))
    if os.path.exists(PATHOUT) is not True:
        os.makedirs(PATHOUT)

    # Save result
    # TODO February 12, 2023: change this for view function
    label_transform = np.tile(label.sum(axis=0, keepdims=True).transpose((1, 2, 0)), (1, 1, 3))
    print(label_transform.shape)

    img_result = np.concatenate(
        (
            img[:, :, (2, 1, 0)],
            label_transform * 255,
        ),
        axis=1,
    )
    cv2.imwrite(PATHNAME, img_result)

    print("DONE!!!")


if __name__ == "__main__":
    pytest.main([__file__])
