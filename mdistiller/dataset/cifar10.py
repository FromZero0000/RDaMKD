import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


class CIFAR10Instance(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_cifar10_train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def get_cifar10_test_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def get_cifar10_dataloaders(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_set = CIFAR10Instance(
        root=data_folder,
        download=True,
        train=True,
        transform=get_cifar10_train_transform(),
    )
    test_set = datasets.CIFAR10(
        root=data_folder,
        download=True,
        train=False,
        transform=get_cifar10_test_transform(),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, len(train_set)
