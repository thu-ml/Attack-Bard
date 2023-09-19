import numpy as np
import os
import random
from PIL import Image
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

"""
Please download data on
https://zenodo.org/record/2535967
"""

__all__ = [
    'get_cifar_10_c_loader',
]

corruptions = [
    'glass_blur',
    'gaussian_noise',
    'shot_noise',
    'speckle_noise',
    'impulse_noise',
    'defocus_blur',
    'gaussian_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    'spatter',
    'saturate',
    'frost',
]


class CIFAR10C(datasets.VisionDataset):
    def __init__(self,
                 name: str,
                 root: str = './resources/CIFAR10-C/',
                 transform=None, target_transform=None):
        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        # if you want to only test a small mount of data, uncomment the following codes
        # self.data = np.concatenate([self.data[0:1000], self.data[10000:11000],
        #                             self.data[20000:21000], self.data[30000:31000],
        #                             self.data[40000:41000]])
        # self.targets = np.concatenate([self.targets[0:1000], self.targets[10000:11000],
        #                                self.targets[20000:21000], self.targets[30000:31000],
        #                                self.targets[40000:41000]])

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset: int, random_subset: bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)


def get_cifar_10_c_loader(name: str = 'gaussian_blur',
                          augment=False,
                          batch_size=128,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=False):
    if not augment:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    set = CIFAR10C(name, transform=transform)
    loader = DataLoader(set, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader
