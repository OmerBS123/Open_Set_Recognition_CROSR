import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision import datasets
from .data_set import OODDataSet, TransformedDataset
from .transformers_utils import get_ood_transformer, get_train_transformer


def get_cifar10_dataset(transform=None, num_samples=1000):
    cifar10_dataset = OODDataSet(datasets.CIFAR10(root='./data', train=False, download=True, transform=transform))
    subset_indices = torch.randperm(len(cifar10_dataset))[:num_samples]
    cifar10_subset = Subset(cifar10_dataset, subset_indices)
    return cifar10_subset


def get_fashion_mnist_dataset(transform=None, num_samples=1000):
    fashion_mnist_dataset = OODDataSet(datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform))
    subset_indices = torch.randperm(len(fashion_mnist_dataset))[:num_samples]
    fashion_mnist_subset = Subset(fashion_mnist_dataset, subset_indices)
    return fashion_mnist_subset


def get_mnist_test_dataset(num_samples=3000):
    # Load MNIST test data
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)

    # Create a subset of the MNIST test data
    if num_samples > len(mnist_test):
        raise ValueError("Requested number of samples exceeds the number of available MNIST test samples")

    indices = torch.randperm(len(mnist_test)).tolist()[:num_samples]
    mnist_subset = Subset(mnist_test, indices)

    return mnist_subset


def get_mnist_train_dataset(limit=10000):
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True)

    # # Create a subset with the first `limit` images
    indices = np.arange(limit)
    mnist_train_subset = Subset(mnist_trainset, indices)

    return mnist_train_subset


def get_mnist_testloader(batch_size=64):
    # Define the transformers
    transform = get_ood_transformer()

    # Load MNIST test dataset
    mnist_dataset = get_mnist_test_dataset(num_samples=4000)

    data_set_transformed = TransformedDataset(dataset=mnist_dataset, transform=transform)

    return DataLoader(data_set_transformed, batch_size=batch_size, shuffle=False, num_workers=2)


def get_combined_testloader(mnist_data_size=3000, cifar_data_size=1000, fashion_data_size=1000, batch_size=64):
    # Define the transformers
    transform = get_ood_transformer()

    # Load MNIST test dataset
    mnist_dataset = get_mnist_test_dataset(num_samples=mnist_data_size)

    cifar10_dataset = get_cifar10_dataset(num_samples=cifar_data_size)

    fashion_mnist_dataset = get_fashion_mnist_dataset(num_samples=fashion_data_size)

    # Combine datasets
    combined_dataset = ConcatDataset([mnist_dataset, cifar10_dataset, fashion_mnist_dataset])
    combined_dataset = TransformedDataset(dataset=combined_dataset, transform=transform)

    # Create DataLoader
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return combined_loader


def get_mnist_cifar10_testloader(mnist_data_size=3000, cifar_data_size=1000, batch_size=64):
    """
    Returns a DataLoader combining MNIST and CIFAR-10 datasets.

    Parameters:
    - mnist_data_size: Number of samples to load from the MNIST dataset.
    - cifar_data_size: Number of samples to load from the CIFAR-10 dataset.
    - batch_size: Batch size for the DataLoader.

    Returns:
    - combined_loader: DataLoader with the combined MNIST and CIFAR-10 datasets.
    """
    # Define the transformers
    transform = get_ood_transformer()

    # Load MNIST test dataset
    mnist_dataset = get_mnist_test_dataset(num_samples=mnist_data_size)

    # Load CIFAR-10 dataset
    cifar10_dataset = get_cifar10_dataset(num_samples=cifar_data_size)

    # Combine datasets
    combined_dataset = ConcatDataset([mnist_dataset, cifar10_dataset])
    combined_dataset = TransformedDataset(dataset=combined_dataset, transform=transform)

    # Create DataLoader
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return combined_loader


def get_mnist_fashionmnist_testloader(mnist_data_size=3000, fashion_data_size=1000, batch_size=64):
    """
    Returns a DataLoader combining MNIST and FashionMNIST datasets.

    Parameters:
    - mnist_data_size: Number of samples to load from the MNIST dataset.
    - fashion_data_size: Number of samples to load from the FashionMNIST dataset.
    - batch_size: Batch size for the DataLoader.

    Returns:
    - combined_loader: DataLoader with the combined MNIST and FashionMNIST datasets.
    """
    # Define the transformers
    transform = get_ood_transformer()

    # Load MNIST test dataset
    mnist_dataset = get_mnist_test_dataset(num_samples=mnist_data_size)

    # Load FashionMNIST dataset
    fashion_mnist_dataset = get_fashion_mnist_dataset(num_samples=fashion_data_size)

    # Combine datasets
    combined_dataset = ConcatDataset([mnist_dataset, fashion_mnist_dataset])
    combined_dataset = TransformedDataset(dataset=combined_dataset, transform=transform)

    # Create DataLoader
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return combined_loader


def get_mnist_trainloader(batch_size=64):
    # Define the transformers
    transform = get_train_transformer()

    # Load MNIST test dataset
    mnist_train_dataset = get_mnist_train_dataset()
    mnist_train_dataset = TransformedDataset(dataset=mnist_train_dataset, transform=transform)

    # Create DataLoader
    train_loader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader


def get_train_val_loaders_from_fold_indices(train_dataset, ood_dataset, train_idx, mnist_val_idx, ood_val_idx, batch_size=64):
    val_transformer = get_ood_transformer()
    train_transformer = get_train_transformer()
    train_subset = Subset(train_dataset, train_idx)
    mnist_val_subset = Subset(train_dataset, mnist_val_idx)
    ood_val_subset = Subset(ood_dataset, ood_val_idx)
    combined_val_set = ConcatDataset([mnist_val_subset, ood_val_subset])
    val_data_set = TransformedDataset(dataset=combined_val_set, transform=val_transformer)
    train_data_set = TransformedDataset(dataset=train_subset, transform=train_transformer)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
