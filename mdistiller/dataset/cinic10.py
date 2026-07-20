import os
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


class CINIC10CleanInstance(Dataset):
    """CINIC-10 split with CIFAR-origin images removed by manifest."""

    def __init__(self, root, manifest_root, split, transform=None, return_index=True):
        self.root = root
        self.manifest_root = manifest_root
        self.split = split
        self.transform = transform
        self.return_index = return_index
        self.samples = self._load_manifest()

    def _load_manifest(self):
        manifest = os.path.join(self.manifest_root, "{}.txt".format(self.split))
        if not os.path.isfile(manifest):
            raise FileNotFoundError(
                "Missing CINIC-10 clean manifest: {}. Run "
                "`python tools/prepare_cinic10.py --download --extract --manifest` first.".format(
                    manifest
                )
            )
        samples = []
        with open(manifest, "r") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.split("\t")
                samples.append((os.path.join(self.root, rel_path), int(label)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if not self.return_index:
            return img, target
        return img, target, index


class CINIC10CIFAR10PairedInstance(CINIC10CleanInstance):
    """CINIC-10 target samples paired with same-class CIFAR-10 source samples."""

    def __init__(
        self,
        root,
        manifest_root,
        split,
        transform=None,
        cifar_root=None,
        cifar_transform=None,
        return_index=True,
    ):
        super().__init__(
            root=root,
            manifest_root=manifest_root,
            split=split,
            transform=transform,
            return_index=return_index,
        )
        if cifar_root is None:
            cifar_root = get_data_folder()
        self.cifar = datasets.CIFAR10(
            root=cifar_root,
            train=True,
            download=False,
            transform=cifar_transform,
        )
        self.cifar_by_class = [[] for _ in range(len(CLASS_NAMES))]
        for idx, label in enumerate(self.cifar.targets):
            self.cifar_by_class[int(label)].append(idx)

    def __getitem__(self, index):
        img, target, index = super().__getitem__(index)
        source_indices = self.cifar_by_class[int(target)]
        source_index = random.choice(source_indices)
        source_img, source_target = self.cifar[source_index]
        if int(source_target) != int(target):
            raise RuntimeError("CIFAR-10 source label does not match CINIC-10 target label.")
        return img, target, index, source_img


def get_cinic10_train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
        ]
    )


def get_cinic10_test_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
        ]
    )


def get_cifar10_teacher_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_cinic10_clean_dataloaders(
    batch_size, val_batch_size, num_workers, manifest_dir="cinic-10-clean"
):
    data_folder = get_data_folder()
    cinic_root = os.path.join(data_folder, "cinic-10")
    manifest_root = os.path.join(data_folder, manifest_dir)

    train_set = CINIC10CleanInstance(
        root=cinic_root,
        manifest_root=manifest_root,
        split="train",
        transform=get_cinic10_train_transform(),
        return_index=True,
    )
    test_set = CINIC10CleanInstance(
        root=cinic_root,
        manifest_root=manifest_root,
        split="test",
        transform=get_cinic10_test_transform(),
        return_index=False,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, len(train_set)


def get_cinic10_clean_cifar_size_dataloaders(batch_size, val_batch_size, num_workers):
    return get_cinic10_clean_dataloaders(
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        manifest_dir="cinic-10-clean-cifar-size",
    )


def get_cinic10_clean_cifar_size_paired_dataloaders(
    batch_size, val_batch_size, num_workers
):
    data_folder = get_data_folder()
    cinic_root = os.path.join(data_folder, "cinic-10")
    manifest_root = os.path.join(data_folder, "cinic-10-clean-cifar-size")

    train_set = CINIC10CIFAR10PairedInstance(
        root=cinic_root,
        manifest_root=manifest_root,
        split="train",
        transform=get_cinic10_train_transform(),
        cifar_root=data_folder,
        cifar_transform=get_cifar10_teacher_transform(),
        return_index=True,
    )
    test_set = CINIC10CleanInstance(
        root=cinic_root,
        manifest_root=manifest_root,
        split="test",
        transform=get_cinic10_test_transform(),
        return_index=False,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, len(train_set)
