import random

import cv2
import numpy as np
import numpy.matlib as mth
from skimage import color


class Render(object):
    def __init__(self, *args, **kwargs):
        pass

    def istouch(self, centers, radios, c, r):
        if len(centers) == 0:
            return False
        d = np.sum((centers - c) ** 2, axis=1) ** 0.5
        return np.any((d < (r + radios)))

    @staticmethod
    def to_rgb(img):
        img = img.astype(np.float32)
        img[np.isnan(img)] = 0
        img -= np.amin(img)
        img /= np.amax(img)

        blue = np.clip(4 * (0.75 - img), 0, 1)
        red = np.clip(4 * (img - 0.25), 0, 1)
        green = np.clip(4 * np.fabs(img - 0.5) - 1.0, 0, 1)

        rgb = np.stack((red, green, blue), axis=2)
        rgb = (rgb * 255).astype(np.uint8)

        return rgb

    @staticmethod
    def to_noise(img, sigma=0.1):
        H, W = img.shape[:2]
        img = img.astype(np.float32) / 255.0
        noise = np.array([random.gauss(0, sigma) for i in range(H * W)])
        noise = noise.reshape(H, W)
        noisy = img + noise
        noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
        return noisy


class CircleRender(Render):
    def __init__(self):
        pass

    @staticmethod
    def generatecircle(n, m, cnt, rmin, rmax, border, sigma):
        mask = np.zeros((n, m), dtype=np.uint8)
        cx = random.randint(border, m - border)
        cy = random.randint(border, n - border)
        r = random.randint(rmin, rmax)
        h = random.randint(1, 255)
        center = [cx, cy]
        mask = cv2.circle(mask, (cx, cy), r, 1, -1)
        return mask, center, r, h

    @staticmethod
    def generate(n, m, cnt, rmin, rmax, border, sigma, btouch):
        """
        @param n,m dim
        @param cnt
        @param rmin,rmax
        @param border
        @param sigma
        @param btouch

        # Example
        n = 512; m = 512; cnt = 5;
        rmin = 5; rmax = 50;
        border = 90;
        sigma = 20;
        img, label, meta = CircleRender.generate( n, m, cnt, rmin, rmax, border, sigma, true)

        """

        images = np.ones((n, m), dtype=np.uint8)
        labels = np.zeros((cnt, n, m), dtype=bool)
        centers = np.zeros((cnt, 2))
        radios = np.zeros((cnt,))

        k = 0
        for i in range(cnt):
            mask, center, r, h = CircleRender().generatecircle(n, m, cnt, rmin, rmax, border, sigma)
            if btouch and Render().istouch(centers[: k + 1, :], radios[: k + 1], center, r):
                continue

            images[mask == 1] = h
            labels[k, ...] = mask
            centers[i, :] = center
            radios[i] = r
            k += 1

        images = Render().to_noise(images, sigma=sigma)
        images = Render().to_rgb(images)

        metadata = {
            "centers": centers,
            "radios": radios,
        }

        return images, labels, metadata


class EllipseRender(Render):
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def generateellipse(n, m, cnt, rmin, rmax, border, sigma):
        mask = np.zeros((n, m), dtype=np.uint8)
        cx = random.randint(border, m - border)
        cy = random.randint(border, n - border)
        aMenor = random.randint(rmin, rmax)
        aMayor = random.randint(rmin, rmax)
        angle = random.randint(0, 360)

        h = random.randint(1, 255)
        center = [cx, cy]
        axis = [aMenor, aMayor]
        mask = cv2.ellipse(mask, (cx, cy), (aMayor, aMenor), angle, 0, 360, 1, -1)

        return mask, center, axis, h

    @staticmethod
    def generate(n, m, cnt, rmin, rmax, border, sigma, btouch):
        """
        @param n,m dim
        @param cnt
        @param rmin,rmax
        @param border
        @param sigma
        @param btouch

        # Example
        n = 512; m = 512; cnt = 5;
        rmin = 5; rmax = 50;
        border = 90;
        sigma = 20;
        img, label, meta = CircleRender.generate( n, m, cnt, rmin, rmax, border, sigma, true)

        """

        images = np.ones((n, m), dtype=np.uint8)
        labels = np.zeros((cnt, n, m), dtype=bool)
        centers = np.zeros((cnt, 2))
        axiss = np.zeros((cnt, 2))

        k = 0
        for i in range(cnt):
            mask, center, axis, h = EllipseRender().generateellipse(n, m, cnt, rmin, rmax, border, sigma)
            if btouch and Render().istouch(centers[: k + 1, :], axiss[: k + 1, 1], center, axis[1]):
                continue

            images[mask == 1] = h
            labels[k, ...] = mask
            centers[i, :] = center
            axiss[i, :] = axis
            k += 1

        images = Render().to_noise(images, sigma=sigma)
        images = Render().to_rgb(images)

        metadata = {
            "centers": centers,
            "axis": axiss,
        }

        return images, labels, metadata
