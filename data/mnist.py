from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

__all__ = ['get_mnist_test', 'get_mnist_train']


def get_mnist_train(batch_size=256,
                    num_workers=8,
                    pin_memory=True,
                    ):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    set = MNIST('./resources/mnist/', train=True, download=True, transform=transform)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                        shuffle=True)
    return loader


def get_mnist_test(batch_size=256,
                   num_workers=8,
                   pin_memory=True, ):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    set = MNIST('./resources/mnist/', train=False, download=True, transform=transform)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return loader
