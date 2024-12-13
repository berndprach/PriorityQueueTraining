import torchvision

from torch.utils.data import random_split
from torchvision import transforms as tfs

from src.datasets.dataset import Dataset


def get_cifar_10(use_test_set=False) -> Dataset:
    if use_test_set:
        return get_cifar_10_train_test()
    else:
        return get_cifar_10_train_val()


def get_cifar_10_train_val():
    train_val_ds = get_cifar_10_data(train=True)
    train_ds, val_ds = random_split(train_val_ds, [45000, 5000])
    return Dataset(train_ds, val_ds)


def get_cifar_10_train_test():
    train_ds = get_cifar_10_data(train=True)
    test_ds = get_cifar_10_data(train=False)
    return Dataset(train_ds, test_ds)


def get_cifar_10_data(train=True):
    cifar_channel_means = (0.4914, 0.4822, 0.4465)
    center_data = tfs.Normalize(cifar_channel_means, (1., 1., 1.))
    try:
        return torchvision.datasets.CIFAR10(
            root="data",
            train=train,
            transform=tfs.Compose([tfs.ToTensor(), center_data]),
            download=False,
        )
    except RuntimeError:
        return torchvision.datasets.CIFAR10(
            root="data",
            train=train,
            transform=tfs.Compose([tfs.ToTensor(), center_data]),
            download=True,
        )


