from torch.utils.data import Dataset, DataLoader
import csv
import os
from PIL import Image
from torchvision import transforms

__kaggle_link__ = 'kaggle datasets download -d google-brain/nips-2017-adversarial-learning-development-set'


class NIPS17(Dataset):
    def __init__(self, images_path='./resources/NIPS17/images/',
                 label_path='./resources/NIPS17/images.csv',
                 transform=transforms.Compose([
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                 ]),
                 ):
        self.labels = {}
        with open(label_path) as f:
            reader = csv.reader(f)
            for line in list(reader)[1:]:
                name, label = line[0], int(line[6]) - 1
                self.labels[name + '.png'] = label
        self.images = os.listdir(images_path)
        self.images.sort()
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        name = self.images[item]
        x = Image.open(os.path.join(self.images_path, name))
        y = self.labels[name]
        return self.transform(x), y


def get_NIPS17_loader(batch_size=64,
                      num_workers=8,
                      pin_memory=True,
                      download=False,
                      shuffle=False,
                      transform=transforms.Compose([
                          transforms.Resize((224, 224)),
                          transforms.ToTensor(),
                      ]),
                      **kwargs,
                      ):
    if download:
        os.system(__kaggle_link__)
    set = NIPS17(transform=transform, **kwargs)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                        shuffle=shuffle)
    return loader
