from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple
import random
# from robustness.datasets import RestrictedImageNet

__all__ = ['get_imagenet10_loader', 'get_imagenet_loader',
           'get_restricted_imagenet_test_loader']


class ImageNet10(ImageNet):
    def __init__(self,
                 *args,
                 target_class: Tuple[int],
                 maximal_images: int or None = None,
                 **kwargs):
        super(ImageNet10, self).__init__(*args, **kwargs)
        self.target_class = list(target_class)
        result = []
        for x, y in self.samples:
            if y in self.target_class:
                result.append((x, y))
        random.shuffle(result)
        self.maximal_images = maximal_images
        self.samples = result

    def __len__(self):
        if self.maximal_images is not None:
            return self.maximal_images
        return len(self.samples)


def get_transform(augment=False):
    if not augment:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.AutoAugment(transforms.AutoAugmentPolicy),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transform


def get_imagenet_loader(
        root='/cephfs-thu/LargeData/ImageNet/',
        split='val',
        augment=False,
        batch_size=1,
        num_workers=8,
        pin_memory=False,
        shuffle=False,
):
    assert split in ['val', 'train']
    transform = get_transform(augment)
    set = ImageNet(root, split, transform=transform)
    loader = DataLoader(set, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    return loader


def get_imagenet10_loader(
        target_class=(0, 100, 200, 300, 400, 500, 600, 700, 800, 900),
        maximum_images=None,
        root='/cephfs-thu/LargeData/ImageNet/',
        split='val',
        augment=False,
        batch_size=1,
        num_workers=8,
        pin_memory=False,
        shuffle=False,
):
    assert split in ['val', 'train']
    transform = get_transform(augment)
    set = ImageNet10(root, split, target_class=target_class, transform=transform,
                     maximal_images=maximum_images)
    loader = DataLoader(set, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    return loader


def get_restricted_imagenet_test_loader(
        root='/workspace/home/chenhuanran2022/dataset/ImageNet/',
        split='val',
        augment=False,
        batch_size=1,
        num_workers=8,
        shuffle=False
):
    assert split in ['val', 'train']
    transform = get_transform(augment)
    set = RestrictedImageNet(data_path=root)
    _, test_loader = set.make_loaders(workers=num_workers, batch_size=batch_size, shuffle_val=shuffle)
    return test_loader
