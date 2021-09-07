from torch.utils.data import Dataset
import cv2
import os


class ImageFolder(Dataset):
    def __init__(self, root, file_list="train.txt", label_file="label.txt", image_size=None, transform=None):
        self.root = root
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transform = transform

        lines = open(os.path.join(root, file_list)).readlines()
        self.files = [line.strip().split(",") for line in lines if line.strip()]

        self.label_names = open(os.path.join(root, label_file)).read().strip().split(",")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_file, mask_file = self.files[i]

        img = cv2.imread(os.path.join(self.root, img_file))
        mask = cv2.imread(os.path.join(self.root, mask_file), cv2.IMREAD_GRAYSCALE)

        if self.image_size is not None:
            img = cv2.resize(img, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask

    @property
    def num_classes(self):
        return len(self.label_names)


def test():
    import sys
    sys.path.append("D:\\dlab\\face_parsing")

    import numpy as np
    from utils import label_colormap

    colormap = label_colormap(19)
    alpha = 0.8

    dataset = ImageFolder("D:\\face\\parsing\\dataset\\CelebAMask-HQ_processed")
    for i, (img, mask) in enumerate(dataset):
        res = colormap[mask]
        dst = (1 - alpha) * img.astype(float) + alpha * res.astype(float)
        img = np.clip(dst.round(), 0, 255).astype(np.uint8)

        cv2.imwrite(f"output/{i:04}.jpg", img)
        if i > 100:
            break


if __name__ == '__main__':
    test()
