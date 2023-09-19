from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from utils.ImageHandling import save_list_images
import numpy as np
import os


def get_loader(dataset: Dataset,
               batch_size=32,
               shuffle=True,
               ):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def save_dataset(
        x: Tensor,
        y: Tensor,
        path: str,
        gt_saving_name: str,
):
    if not os.path.exists(path):
        os.makedirs(path)
    ground_truth = dict()
    save_list_images(x.split(1, dim=0), folder_path=path)
    for i in range(y.shape[0]):
        ground_truth[str(i) + '.png'] = y[i].item()
    np.save(os.path.join(path, gt_saving_name), ground_truth)
    print(f'Successfully create a new dataset with {len(ground_truth)} images')
