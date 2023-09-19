'''
this file aims to read any dataset satisfied that:
    1.all the images are in one folder
    2.only a dict to store ground truth. Keys are image names, values are ground truth labels.
'''

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


def sort_key(x: str) -> int:
    return int(x[:-4])


class SomeDataSet(Dataset):
    def __init__(self, img_path, gt_path, transform=transforms.ToTensor()):
        self.transform = transform
        self.images = [img for img in os.listdir(img_path) if img.endswith('.jpg') or img.endswith('.png')]
        self.images.sort(key=sort_key)
        self.gt = np.load(gt_path, allow_pickle=True).item()
        self.img_path = img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        now = self.images[item]
        now_img = Image.open(os.path.join(self.img_path, now))  # numpy
        return self.transform(now_img), self.gt[now]


def get_someset_loader(img_path,
                       gt_path,
                       batch_size=256,
                       num_workers=8,
                       pin_memory=False,
                       shuffle=False,
                       transform=transforms.ToTensor(),
                       ):
    set = SomeDataSet(img_path=img_path, gt_path=gt_path, transform=transform)
    loader = DataLoader(set, batch_size=batch_size,
                        num_workers=num_workers, shuffle=shuffle,
                        pin_memory=pin_memory)
    return loader
