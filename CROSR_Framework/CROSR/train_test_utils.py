# imports
import os

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.optim import SGD
from tqdm import tqdm

from .data_utils import get_train_val_loaders_from_fold_indices, get_mnist_trainloader, get_mnist_train_dataset, get_fashion_mnist_dataset, get_combined_testloader
from .dhr_nets import DHRNet
from .plot_utils import plot_loss_over_epochs_train, plot_loss_over_epochs_val, plot_validation_accuracy, PLOTS_DIR_NAME, plot_validation_accuracy_f_measures
from .weibull_distribution_utils import compute_weibull_disterbution, set_tail_size


def epoch_train(net, train_loader, optimizer, device):
    """
    Train the network for one epoch using the provided train_loader and optimizer.

    Parameters:
    - net: The neural network model to be trained.
    - train_loader: DataLoader containing the training data.
    - optimizer: Optimizer used to update the model parameters.
    - device: The device to run the training on (e.g., 'cpu' or 'cuda').

    Returns:
    - A list containing the following metrics:
        - Training accuracy.
        - Average cross-entropy loss.
        - Average reconstruction loss.
        - Average total loss.
    """

    net.train()  # Set the model to training mode
    correct = 0
    total = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconstruct_loss = 0.0
    iter = 0
    cls_criterion = nn.CrossEntropyLoss()
    reconstruct_criterion = nn.MSELoss()
    logits_acc, latent_acc, labels_acc = [], [], []

    for inputs, labels in train_loader:
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        logits, reconstruct, latent_representations = net(inputs)

        # Compute losses
        cls_loss = cls_criterion(logits, labels)
        reconstruct_loss = reconstruct_criterion(reconstruct, inputs)
        loss = cls_loss + reconstruct_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reconstruct_loss += reconstruct_loss.item()

        # Compute accuracy
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iter += 1

        # Concatenate and flatten latent representations
        latent_representation_concat = torch.cat(latent_representations, dim=1)  # Shape: [64, 96, 1, 1]
        latent_representation_flat = latent_representation_concat.view(latent_representation_concat.shape[0], -1)  # Shape: [64, 96]

        # Accumulate logits, latent representations, and labels
        logits_acc.append(logits)
        latent_acc.append(latent_representation_flat)
        labels_acc.append(labels)

    # Concatenate accumulated logits, latent representations, and labels
    logits_acc = torch.cat(logits_acc, dim=0)
    latent_acc = torch.cat(latent_acc, dim=0)
    labels_acc = torch.cat(labels_acc)
    logits_latents_cat_acc = torch.cat((logits_acc, latent_acc), dim=1)

    # Compute Weibull distribution parameters
    weibull_params = compute_weibull_disterbution(logits_latents_cat_acc, labels_acc, logits_acc)

    # Set Weibull parameters in the network
    net.set_weibull_params(weibull_params)

    # Return metrics
    return [
        100 * (correct / total),  # Training accuracy
        total_cls_loss / iter,  # Average cross-entropy loss
        total_reconstruct_loss / iter,  # Average reconstruction loss
        total_loss / iter  # Average total loss
    ]


def epoch_val(net, val_loader):
    net.eval()

    correct = 0
    total = 0
    total_loss = 0.0
    total_iter = 0
    correct_in_dist = 0
    total_in_dist = 0
    correct_ood = 0
    total_ood = 0
    cls_criterion = nn.CrossEntropyLoss()
    device = get_device()
    logits_acc, labels_acc = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            logits = net(inputs)

            loss = cls_criterion(logits, labels)

            total_loss = total_loss + loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            total_iter += 1

            # Create boolean masks
            is_ood = (labels == 10)
            is_in_dist = (labels != 10)

            total_ood += is_ood.sum().item()
            total_in_dist += is_in_dist.sum().item()

            # Calculate correct and incorrect predictions for OOD and in-distribution
            correct_preds = (predicted == labels)
            correct += correct_preds.sum().item()
            correct_ood += (correct_preds & is_ood).sum().item()
            correct_in_dist += (correct_preds & is_in_dist).sum().item()

            total_iter += 1

            logits_acc.append(logits)
            labels_acc.append(labels)

        total_acc = (100 * (correct / total))
        in_dist_acc = 100 * (correct_in_dist / total_in_dist)
        ood_acc = 100 * (correct_ood / total_ood)

        logits_acc = torch.cat(logits_acc, dim=0)
        labels_acc = torch.cat(labels_acc)

        f_ma, f_mi = calculate_osr_f1(labels_acc, logits_acc, 10)

    return total_acc, (total_loss / total_iter), in_dist_acc, ood_acc, f_ma, f_mi


def train_net(net, train_loader, device, num_epochs=15, lr=0.001, momentum=0.9, weight_decay=0.0001):
    """
    Train the given neural network on the provided training data.

    Parameters:
    - net: The neural network model to be trained.
    - train_loader: DataLoader containing the training data.
    - device: The device to run the training on (e.g., 'cpu' or 'cuda').
    - num_epochs: Number of epochs to train the model.
    - lr: Learning rate for the optimizer.
    - momentum: Momentum for the SGD optimizer.
    - weight_decay: Weight decay (L2 penalty) for the optimizer.

    Returns:
    - net: The trained model.
    - training_metrics: A tuple containing lists of training accuracies, cross-entropy losses, reconstruction losses, and total losses over the epochs.
    """

    # Move the model to the specified device
    net.to(device)

    # Set the model to training mode
    net.train()

    # Initialize the optimizer with the given parameters
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Initialize lists to track metrics over epochs
    train_accuracies = []
    cross_entropy_losses = []
    reconstruction_losses = []
    total_losses = []

    # Training loop over epochs
    for epoch in tqdm(range(num_epochs), desc='Training Epochs', unit='epoch'):
        epoch_train_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')

        # Training for the current epoch
        train_acc, cls_acc, reconstruct_acc, total_loss = epoch_train(net, epoch_train_loader, optimizer, device)

        # Record metrics
        train_accuracies.append(train_acc)
        cross_entropy_losses.append(cls_acc)
        reconstruction_losses.append(reconstruct_acc)
        total_losses.append(total_loss)

    return net, (train_accuracies, cross_entropy_losses, reconstruction_losses, total_losses)


def train_net_with_val(net, train_loader, val_loader, num_epochs=30, lr=0.001, momentum=0.9, weight_decay=0.0001):
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_accuracies = []
    cross_entropy_losses = []
    reconstruction_losses = []
    total_losses = []

    val_accuracies = []
    val_total_losses = []
    in_distribution_accuracies = []
    ood_accuracies = []
    last_epoch_val_accuracy = 0
    last_epoch_in_dist_accuracy = 0
    last_epoch_ood_accuracy = 0

    for epoch in tqdm(range(num_epochs), desc='Training Epochs', unit='epoch'):
        epoch_train_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        train_acc, cls_acc, reconstruct_acc, total_loss = epoch_train(net, epoch_train_loader, optimizer)

        train_accuracies.append(train_acc)
        cross_entropy_losses.append(cls_acc)
        reconstruction_losses.append(reconstruct_acc)
        total_losses.append(total_loss)

        with torch.no_grad():
            print(f'Starting val for epoch {epoch}')
            val_accuracy, total_val_loss, in_distribution_accuracy, ood_accuracy, _, _ = epoch_val(net, val_loader)

            val_accuracies.append(val_accuracy)
            val_total_losses.append(total_val_loss)
            in_distribution_accuracies.append(in_distribution_accuracy)
            ood_accuracies.append(ood_accuracy)

        if epoch == num_epochs - 1:
            last_epoch_val_accuracy = val_accuracy
            last_epoch_in_dist_accuracy = in_distribution_accuracy
            last_epoch_ood_accuracy = ood_accuracy

    return (train_accuracies, cross_entropy_losses, reconstruction_losses, total_losses), (val_accuracies, val_total_losses, in_distribution_accuracies, ood_accuracies), (last_epoch_val_accuracy, last_epoch_in_dist_accuracy, last_epoch_ood_accuracy)


def k_fold_validation_tail_size(train_dataset, ood_dataset, tail_size, num_folds=3, do_plot=True):
    kf = KFold(n_splits=num_folds, shuffle=True)
    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(ood_dataset)))

    val_scores = []
    val_in_distribution_accuracies = []
    val_ood_accuracies = []
    fold_num = 1
    device = get_device()
    set_tail_size(tail_size)
    for (train_idx, mnist_val_idx), (_, ood_val_idx) in zip(kf.split(train_indices), kf.split(val_indices)):
        print(f'Started fold num:{fold_num} for tail size:{tail_size}')
        train_loader, val_loader = get_train_val_loaders_from_fold_indices(train_dataset, ood_dataset, train_idx, mnist_val_idx, ood_val_idx)

        # create new net
        net = DHRNet()
        net = net.to(device)

        train_data_for_plot, val_data_for_plot, last_epoch_accuracy = train_net_with_val(net, train_loader, val_loader, num_epochs=15)
        train_accuracies, cross_entropy_losses, reconstruction_losses, total_losses = train_data_for_plot
        val_accuracies, _, in_distribution_accuracies, ood_accuracies = val_data_for_plot

        if do_plot:
            plot_loss_over_epochs_train(train_acc=train_accuracies, cls_loss=cross_entropy_losses, reconstruct_loss=reconstruction_losses, total_loss=total_losses, file_name_to_save=f'tail_size_{tail_size}/fold_num_{fold_num}/train_losses', title=f'Train loss over epochs', display=False)
            plot_loss_over_epochs_val(val_accuracy=val_accuracies, in_distribution_acc=in_distribution_accuracies, ood_accuracy=ood_accuracies, file_name_to_save=f'tail_size_{tail_size}/fold_num_{fold_num}/val_losses', title=f'Val loss over epochs for tail size {tail_size}', display=False)

        last_epoch_val_acc, last_epoch_in_dist_acc, last_epoch_ood_acc = last_epoch_accuracy

        val_scores.append(last_epoch_val_acc)
        val_in_distribution_accuracies.append(last_epoch_in_dist_acc)
        val_ood_accuracies.append(last_epoch_ood_acc)

        print(f'Finished fold num:{fold_num} for tail size:{tail_size}')

        fold_num += 1

    avg_val_score = sum(val_scores) / num_folds
    avg_in_dist_score = sum(val_in_distribution_accuracies) / num_folds
    avg_ood_score = sum(val_ood_accuracies) / num_folds

    return avg_val_score, avg_in_dist_score, avg_ood_score


def cross_validation_for_tail_size(tail_sizes):
    print('Starting cross validation')
    train_dataset = get_mnist_train_dataset()
    # ood_dataset = get_cifar_and_fashion_model_dataset()
    ood_dataset = get_fashion_mnist_dataset()
    accuracy_dict_numbers = dict()
    accuracy_dict_f_measures = dict()

    train_loader = get_mnist_trainloader()
    test_loader = get_combined_testloader()
    device = get_device()

    for tail_size in tail_sizes:
        print(f'Running kfold for tail_size:{tail_size}')
        avg_val_score, avg_in_dist_score, avg_ood_score = k_fold_validation_tail_size(train_dataset, ood_dataset, tail_size, num_folds=2, do_plot=True)
        accuracy_dict_numbers[tail_size] = (avg_val_score, avg_in_dist_score, avg_ood_score)
        print(f'Finished K-fold for threshold:{tail_size}')
        print(f'Training and testing net for tail size: {tail_size}')
        net = DHRNet()
        net = net.to(device)
        net, _ = train_net(net, train_loader, num_epochs=15, lr=0.001, momentum=0.9, weight_decay=0.0001)
        f_ma, f_mi = evaluate_model_ood(net, test_loader)
        accuracy_dict_f_measures[tail_size] = (f_ma, f_mi)

    print(f'The accuracy dict numbers is:{accuracy_dict_numbers}')
    print(f'The accuracy dict f measures is:{accuracy_dict_numbers}')
    curr_dir = os.path.dirname(__file__)
    accuracy_plot_numbers_path_to_save = os.path.join(curr_dir, PLOTS_DIR_NAME, 'cross_validation_accuracy_bars_numbers') + '.png'
    accuracy_plot_f_measures_path_to_save = os.path.join(curr_dir, PLOTS_DIR_NAME, 'cross_validation_accuracy_bars_f_measures') + '.png'
    plot_validation_accuracy(accuracy_dict_numbers, save_path=accuracy_plot_numbers_path_to_save, display=False)
    plot_validation_accuracy_f_measures(accuracy_dict_f_measures, save_path=accuracy_plot_f_measures_path_to_save, display=False)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    # elif os.uname().sysname == 'Darwin':
    #     return 'mps'
    return 'cpu'


def evaluate_model_ood(model, test_loader):
    model.eval()
    total_accuracy, _, in_distribution_accuracy, ood_accuracy, f_ma, f_mi = epoch_val(model, test_loader)

    print(f'The total model accuracy is:{total_accuracy:2f}\nThe in dist accuracy is:{in_distribution_accuracy:2f}\nThe ood accuracy is:{ood_accuracy:2f}\nThe f_ma score is:{f_ma:2f}\nThe f_mi score is:{f_mi}')

    return f_ma, f_mi


def calculate_osr_f1(true_labels: torch.Tensor, predicted_labels, num_classes):
    """
    Calculate Macro and Micro F1-Measure for Open Set Recognition.

    Parameters:
    - true_labels: Ground truth labels, tensor of shape (num_samples,)
    - predicted_labels: Predicted labels, tensor of shape (num_samples,)
    - num_classes: Number of KKCs (Known Known Classes)

    Returns:
    - F_ma: Macro F1-Measure
    - F_mi: Micro F1-Measure
    """
    # Initialize counts
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)

    predicted_labels = torch.argmax(predicted_labels, dim=1)
    # True label is within known classes, and predicted is within known classes
    for i in range(num_classes):
        TP[i] = torch.sum((predicted_labels == i) & (true_labels == i)).item()
        FP[i] = torch.sum((predicted_labels == i) & (true_labels != i) & (true_labels < num_classes)).item()
        FN[i] = torch.sum((predicted_labels != i) & (true_labels == i)).item()

    # Misclassifications where true label is unknown but predicted label is known
    FP_UUC = torch.sum((true_labels >= num_classes) & (predicted_labels < num_classes)).item()

    # Misclassifications where true label is known but predicted label is unknown
    FN_UUC = torch.sum((true_labels < num_classes) & (predicted_labels >= num_classes)).item()

    # Update FP and FN with UUC contributions
    FP += FP_UUC
    FN += FN_UUC

    # Macro Precision and Recall
    P_ma = torch.mean(TP / (TP + FP + 1e-10))  # Add a small value to avoid division by zero
    R_ma = torch.mean(TP / (TP + FN + 1e-10))

    # Macro F1-Measure
    F_ma = 2 * P_ma * R_ma / (P_ma + R_ma + 1e-10)

    # Micro Precision and Recall
    P_mi = torch.sum(TP) / (torch.sum(TP) + torch.sum(FP) + 1e-10)
    R_mi = torch.sum(TP) / (torch.sum(TP) + torch.sum(FN) + 1e-10)

    # Micro F1-Measure
    F_mi = 2 * P_mi * R_mi / (P_mi + R_mi + 1e-10)

    return F_ma.item(), F_mi.item()


def train_basline_model(model, train_loader, device, num_epochs=10, learning_rate=0.01):
    """
    Train the given model using the provided train_loader and device.

    Parameters:
    - model: The neural network model to be trained (instance of BaselineNet).
    - train_loader: PyTorch DataLoader containing the training data.
    - device: The device to run the training on (e.g., 'cpu' or 'cuda').
    - num_epochs: Number of epochs to train the model.
    - learning_rate: Learning rate for the optimizer.

    Returns:
    - model: The trained model.
    - loss_history: List of loss values over the epochs.
    - accuracy_history: List of accuracy values over the epochs.
    """

    # Move model to the specified device
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Track the loss and accuracy over epochs
    loss_history = []
    accuracy_history = []

    # Training loop
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for i, (inputs, labels) in enumerate(epoch_iterator):
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # Update tqdm progress bar description
            epoch_iterator.set_postfix(loss=loss.item())

        # Calculate and record the average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return model, (loss_history, accuracy_history)


if __name__ == '__main__':
    # tail_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400]
    # cross_validation_for_tail_size(tail_sizes)
    # curr_dir = os.path.dirname(__file__)
    # accuracy_dict_numbers = {50: (9.975, 4.220000000000001, 96.3), 100: (14.90625, 9.493333333333334, 96.1), 150: (17.018749999999997, 12.16, 89.9), 200: (23.174999999999997, 18.71333333333333, 90.1), 250: (20.36875, 15.746666666666666, 89.7), 300: (27.268749999999997, 23.8, 79.3), 350: (34.125, 32.06666666666667, 65.0),
    #                          400: (35.712500000000006, 33.446666666666665, 69.69999999999999), 450: (39.71875, 38.093333333333334, 64.1), 500: (40.943749999999994, 39.86666666666667, 57.1), 550: (44.90625, 44.21333333333334, 55.300000000000004), 600: (49.725, 49.88, 47.39999999999999), 650: (53.775000000000006, 54.69333333333333, 40.0),
    #                          700: (48.925, 48.67333333333333, 52.7), 750: (46.7625, 46.733333333333334, 47.2), 800: (49.099999999999994, 49.06666666666666, 49.599999999999994), 850: (47.63125, 47.873333333333335, 44.0), 900: (50.6375, 51.06666666666666, 44.2), 950: (43.20625, 42.86, 48.400000000000006), 1000: (48.2875, 48.25333333333333, 48.8),
    #                          1050: (48.3, 48.43333333333334, 46.3), 1100: (50.7625, 50.43333333333334, 55.7), 1150: (48.60625, 48.83333333333333, 45.2), 1200: (50.4625, 50.95333333333333, 43.1), 1250: (49.2625, 49.126666666666665, 51.3), 1300: (49.55, 49.60666666666667, 48.7), 1350: (51.1125, 51.12, 51.0),
    #                          1400: (51.837500000000006, 52.040000000000006, 48.8)}
    # accuracy_dict_f_measures = {50: (9.975, 4.220000000000001, 96.3), 100: (14.90625, 9.493333333333334, 96.1), 150: (17.018749999999997, 12.16, 89.9), 200: (23.174999999999997, 18.71333333333333, 90.1), 250: (20.36875, 15.746666666666666, 89.7), 300: (27.268749999999997, 23.8, 79.3), 350: (34.125, 32.06666666666667, 65.0),
    #                             400: (35.712500000000006, 33.446666666666665, 69.69999999999999), 450: (39.71875, 38.093333333333334, 64.1), 500: (40.943749999999994, 39.86666666666667, 57.1), 550: (44.90625, 44.21333333333334, 55.300000000000004), 600: (49.725, 49.88, 47.39999999999999), 650: (53.775000000000006, 54.69333333333333, 40.0),
    #                             700: (48.925, 48.67333333333333, 52.7), 750: (46.7625, 46.733333333333334, 47.2), 800: (49.099999999999994, 49.06666666666666, 49.599999999999994), 850: (47.63125, 47.873333333333335, 44.0), 900: (50.6375, 51.06666666666666, 44.2), 950: (43.20625, 42.86, 48.400000000000006), 1000: (48.2875, 48.25333333333333, 48.8),
    #                             1050: (48.3, 48.43333333333334, 46.3), 1100: (50.7625, 50.43333333333334, 55.7), 1150: (48.60625, 48.83333333333333, 45.2), 1200: (50.4625, 50.95333333333333, 43.1), 1250: (49.2625, 49.126666666666665, 51.3), 1300: (49.55, 49.60666666666667, 48.7), 1350: (51.1125, 51.12, 51.0),
    #                             1400: (51.837500000000006, 52.040000000000006, 48.8)}
    # accuracy_plot_numbers_path_to_save = os.path.join(curr_dir, PLOTS_DIR_NAME, 'cross_validation_accuracy_bars_numbers') + '.png'
    # accuracy_plot_f_measures_path_to_save = os.path.join(curr_dir, PLOTS_DIR_NAME, 'cross_validation_accuracy_bars_f_measures') + '.png'
    # plot_validation_accuracy(accuracy_dict_numbers, save_path=accuracy_plot_numbers_path_to_save, display=False)
    # plot_validation_accuracy_f_measures(accuracy_dict_f_measures, save_path=accuracy_plot_f_measures_path_to_save, display=False)
    osr_model = DHRNet()
    mnist_trainloader = get_mnist_trainloader()
    device = get_device()
    osr_model, accuracy_and_losses = train_net(osr_model, mnist_trainloader, device, num_epochs=10)
    osr_model_train_accuracies, osr_model_cross_entropy_losses, osr_model_reconstruction_losses, osr_model_total_losses = accuracy_and_losses


