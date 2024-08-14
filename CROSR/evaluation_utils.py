import torch
from sklearn.metrics import confusion_matrix, accuracy_score


def compute_osr_metrics_binary(all_preds, all_labels, known_class_range=(0, 9)):
    """
    Computes the OSR metrics including confusion matrix, total accuracy, in-distribution accuracy, and OOD accuracy.

    Parameters:
    - model: The trained model to evaluate.
    - test_loader: DataLoader containing the test data.
    - known_class_range: Tuple indicating the range of known classes (inclusive).

    Returns:
    - cm: Confusion matrix of shape (2, 2) for binary OSR classification.
    - total_accuracy: Overall accuracy across both known and unknown classes.
    - in_dist_accuracy: Accuracy within the known (in-distribution) classes.
    - ood_accuracy: Accuracy within the unknown (OOD) classes.
    """

    # Map MNIST classes (known) to 0, and OOD classes to 1
    true_binary_labels = [(0 if label in range(known_class_range[0], known_class_range[1] + 1) else 1) for label in all_labels]
    predicted_binary_labels = [(0 if label in range(known_class_range[0], known_class_range[1] + 1) else 1) for label in all_preds]

    # Compute confusion matrix
    cm = confusion_matrix(true_binary_labels, predicted_binary_labels)

    # Calculate in-distribution accuracy (accuracy for known classes)
    in_dist_accuracy = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0

    # Calculate OOD accuracy (accuracy for unknown classes)
    ood_accuracy = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0

    return cm, in_dist_accuracy, ood_accuracy


def compute_osr_confusion_matrix_11_classes(all_preds, all_labels, known_class_range=(0, 9), unknown_class_label=10):
    """
    Computes the confusion matrix for OSR with 11 classes (10 MNIST classes + 1 Unknown class).

    Parameters:
    - model: The trained model to evaluate.
    - test_loader: DataLoader containing the test data.
    - known_class_range: Tuple indicating the range of known classes (inclusive).
    - unknown_class_label: Label assigned to the "Unknown" class (default is 10).

    Returns:
    - cm: Confusion matrix of shape (11, 11).
    """

    # Map MNIST classes to themselves and OOD classes to 'unknown_class_label'
    true_mapped_labels = [label if label in range(known_class_range[0], known_class_range[1] + 1) else unknown_class_label for label in all_labels]
    predicted_mapped_labels = [label if label in range(known_class_range[0], known_class_range[1] + 1) else unknown_class_label for label in all_preds]

    # Compute confusion matrix
    cm = confusion_matrix(true_mapped_labels, predicted_mapped_labels, labels=list(range(known_class_range[0], known_class_range[1] + 1)) + [unknown_class_label])

    return cm


def evaluate_model_on_mnist(model, test_loader, device):
    """
    Evaluate the trained model on the MNIST test set using the specified device.

    Parameters:
    - model: The trained neural network model.
    - test_loader: PyTorch DataLoader containing the MNIST test data.
    - device: The device to run the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
    - accuracy: The accuracy of the model on the MNIST test set.
    - cm: The confusion matrix of the model's predictions on the MNIST test set.
    """

    # Move the model to the specified device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to hold true labels and predictions
    all_preds = []
    all_labels = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm


def evaluate_osr_model(model, test_loader, device):
    """
    Evaluate the OSR model on the specified device, returning true labels, predicted labels, and overall model accuracy.

    Parameters:
    - model: The trained OSR model to evaluate.
    - test_loader: DataLoader containing the test data.
    - device: The device to run the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
    - true_labels: List of true labels for the test set.
    - predicted_labels: List of predicted labels from the model.
    - overall_accuracy: The overall accuracy of the model on the test set.
    """

    # Move the model to the specified device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to hold true labels and predictions
    all_preds = []
    all_labels = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)

    return all_labels, all_preds, overall_accuracy
