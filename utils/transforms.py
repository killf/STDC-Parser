import cv2
import torch
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ToTensor:
    def __call__(self, img, mask):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).type(torch.float32) / 255
        mask = torch.from_numpy(mask).type(torch.int64)
        return img, mask


class RandomResizeRotation:
    def __init__(self, degrees=(-30, 30), center=(0.5, 0.5), scales=(0.75, 1.25)):
        self.degrees = degrees
        self.center = center
        self.scales = scales

    def __call__(self, img, mask):
        h, w = mask.shape

        degree = np.random.randint(self.degrees[0], self.degrees[1])
        scale = np.random.rand() * (self.scales[1] - self.scales[0]) + self.scales[0]

        cx = np.random.randint(int(w * (0.5 - self.center[0] / 2)), int(w * (0.5 + self.center[0] / 2)))
        cy = np.random.randint(int(h * (0.5 - self.center[1] / 2)), int(h * (0.5 + self.center[1] / 2)))

        M = cv2.getRotationMatrix2D((cx, cy), degree, scale)

        img = cv2.warpAffine(img, M, (h, w))
        mask = cv2.warpAffine(mask, M, (h, w))
        return img, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        return img, mask
