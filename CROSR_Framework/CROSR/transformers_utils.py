from torchvision import transforms


def get_ood_transformer():
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalization for grayscale images
    ])


def get_train_transformer():
    return transforms.Compose([
        transforms.RandomApply([
            transforms.RandomRotation(degrees=15),  # Randomly rotate images
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian blur
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0)),  # Randomly resize and crop images
        ], p=0.5),  # Apply the augmentations with a probability of 0.5
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalization for grayscale images
    ])


def get_mnist_transformer():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalization for grayscale images
    ])
