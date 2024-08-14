import os
import torch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import ConfusionMatrixDisplay

PLOTS_DIR_NAME = 'Plots'
curr_dir = os.path.dirname(__file__)
if not os.path.exists(os.path.join(curr_dir, PLOTS_DIR_NAME)):
    os.mkdir(os.path.join(curr_dir, PLOTS_DIR_NAME))


def plot_loss_over_epochs_train(train_acc, cls_loss, reconstruct_loss, total_loss, file_name_to_save='', title='Training Accuracy', display=True):
    path_to_save = ''

    if file_name_to_save:
        curr_dir = os.path.dirname(__file__)
        path_to_save = os.path.join(curr_dir, PLOTS_DIR_NAME, file_name_to_save) + ".png"
        # Extract the directory path
        dir_path = os.path.dirname(path_to_save)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

    epochs = range(1, len(train_acc) + 1)

    fig = plt.figure(figsize=(15, 10))

    # Plot training accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_acc, 'b', label='Total Training Accuracy')
    plt.plot(epochs, cls_loss, 'r', label='Classification error')
    plt.plot(epochs, reconstruct_loss, 'g', label='Reconstruction error')
    plt.plot(epochs, total_loss, 'k', label='Total Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    if path_to_save:
        plt.savefig(path_to_save)

    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_loss_over_epochs_val(val_accuracy, in_distribution_acc, ood_accuracy, file_name_to_save='', title='Validation Accuracy over Epochs', display=True):
    path_to_save = ''

    if file_name_to_save:
        curr_dir = os.path.dirname(__file__)
        path_to_save = os.path.join(curr_dir, PLOTS_DIR_NAME, file_name_to_save) + ".png"

        # Extract the directory path
        dir_path = os.path.dirname(path_to_save)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

    epochs = range(1, len(val_accuracy) + 1)

    fig = plt.figure(figsize=(12, 8))

    # Plot validation accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.plot(epochs, in_distribution_acc, 'g', label='In-distribution Accuracy')
    plt.plot(epochs, ood_accuracy, 'c', label='OOD Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    if path_to_save:
        plt.savefig(path_to_save)

    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_accuracies_cross_validation(f_ma_accuracies, f_mi_accuracies, file_name_to_save='', title='F ma and F mi scores over epochs', display=True):
    path_to_save = ''

    if file_name_to_save:
        curr_dir = os.path.dirname(__file__)
        path_to_save = os.path.join(curr_dir, PLOTS_DIR_NAME, file_name_to_save) + ".png"

        # Extract the directory path
        dir_path = os.path.dirname(path_to_save)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

    epochs = range(1, len(f_ma_accuracies) + 1)

    fig = plt.figure(figsize=(8, 6))

    # Plot validation accuracy
    plt.plot(epochs, f_ma_accuracies, 'b', label='F ma score')
    plt.plot(epochs, f_mi_accuracies, 'r', label='F mi score')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    if path_to_save:
        plt.savefig(path_to_save)

    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_loss_bar_cross_validation(val_avrg_accuracy, hyper_parameters_values, hyper_parameter_name):
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(hyper_parameters_values, val_avrg_accuracy, color='red')
    plt.xlabel(hyper_parameter_name)
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss per {hyper_parameter_name}')
    plt.show()


def plot_validation_accuracy(accuracy_dict, save_path=None, display=True):
    """
    Plot total, in-distribution, and OOD validation accuracies by thresholds using dictionary keys as x-axis labels.

    Parameters:
    - accuracy_dict: Dictionary containing thresholds as keys and a tuple of (total_accuracy, in_dist_accuracy, ood_accuracy) as values.
    - save_path: Optional. Path to save the plot image.
    - display: Optional. If True, displays the plot. If False, closes the plot after saving.
    """

    # Extract keys and values
    keys = list(accuracy_dict.keys())
    total_errors = [accuracy_dict[key][0] for key in keys]
    in_dist_errors = [accuracy_dict[key][1] for key in keys]
    ood_errors = [accuracy_dict[key][2] for key in keys]

    # Set up the plot
    x = np.arange(len(keys))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 10))
    rects1 = ax.bar(x - width, total_errors, width, label='Total Val Accuracy')
    rects2 = ax.bar(x, in_dist_errors, width, label='In-Dist Val Accuracy')
    rects3 = ax.bar(x + width, ood_errors, width, label='OOD Val Accuracy')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('tail sizes')
    ax.set_ylabel('Accuracies')
    ax.set_title('Validation Accuracies by tail sizes')
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.legend()

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save the plot to the specified file path if provided
    if save_path:
        # Extract the directory path
        dir_path = os.path.dirname(save_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path)

    # Display the plot
    plt.tight_layout()
    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_validation_accuracy_f_measures(accuracy_dict, save_path=None, display=True):
    """
    Plot F_ma and F_mi scores by tail size using keys of the dictionary as x-axis labels.

    Parameters:
    - accuracy_dict: Dictionary containing tail sizes as keys and a tuple of (F_ma, F_mi) scores as values.
    - save_path: Optional. Path to save the plot image.
    - display: Optional. If True, displays the plot. If False, closes the plot after saving.
    """

    # Extract keys and values from the dictionary
    list_tail_values = list(accuracy_dict.keys())
    f_ma_scores = [accuracy_dict[key][0] for key in list_tail_values]
    f_mi_scores = [accuracy_dict[key][1] for key in list_tail_values]

    # Set up the plot
    x = np.arange(len(list_tail_values))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 10))
    rects1 = ax.bar(x - width / 2, f_ma_scores, width, label='F ma score')
    rects2 = ax.bar(x + width / 2, f_mi_scores, width, label='F mi score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Tail size')
    ax.set_ylabel('Scores')
    ax.set_title('F scores by tail size')
    ax.set_xticks(x)
    ax.set_xticklabels(list_tail_values)
    ax.legend()

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save the plot to the specified file path if provided
    if save_path:
        # Extract the directory path
        dir_path = os.path.dirname(save_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path)

    # Display the plot
    plt.tight_layout()
    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_circular_classification_v2(data_loader, predicted_labels, true_labels, num_classes=10):
    """
    Plot images in a circular layout with a smaller radius for in-distribution classes
    and a larger radius for OOD classifications.

    Parameters:
    - data_loader: PyTorch DataLoader containing the test data.
    - predicted_labels: Array of predicted labels.
    - true_labels: Array of true labels.
    - num_classes: Number of in-distribution classes (default is 10 for MNIST).
    """
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    # Extract images from the data loader
    images = []
    for inputs, labels in data_loader:
        images.extend(inputs.cpu().numpy())
    images = np.array(images)

    # Set up radii for different groups
    inner_radius = 6.0
    mid_radius = 10.0
    outer_radius = 12.0

    # Angles for classes 0-4 (inner radius)
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)

    # Angles for classes 5-9 (mid radius)
    mid_angles = np.linspace(np.pi / 12, 2 * np.pi + np.pi / 12, 5, endpoint=False)

    fig, ax = plt.subplots(figsize=(15, 15))
    fig.patch.set_facecolor('black')  # Set the figure background to black
    ax.set_facecolor('black')  # Set the axes background to black
    ood_indicies = np.where(true_labels == num_classes)[0]
    # Plot images for classes 0-4 (inner radius)
    for i, angle in enumerate(inner_angles):
        true_class_indices = np.where(true_labels == i)[0]
        labeld_class_indices = np.where(predicted_labels == i)[0]
        correctly_classified = np.intersect1d(true_class_indices, labeld_class_indices)
        incorrectly_classified = np.intersect1d(ood_indicies, labeld_class_indices)
        # incorrectly_classified = class_indices[(true_labels[class_indices] == num_classes)]

        # Select 5 correct samples for each class (MNIST)
        correctly_classified = correctly_classified[:10]
        incorrectly_classified = incorrectly_classified[:2]

        # Scatter points for correct classifications
        for j, idx in enumerate(correctly_classified):
            radius_variation = np.random.uniform(inner_radius - 0.2, inner_radius + 0.2)
            x_pos = radius_variation * np.cos(angle + j * 0.4)
            y_pos = radius_variation * np.sin(angle + j * 0.4)
            img = np.squeeze(images[idx].transpose(1, 2, 0))
            ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        # Scatter points for incorrect classifications (OOD classified as MNIST)
        for j, idx in enumerate(incorrectly_classified):
            radius_variation = np.random.uniform(inner_radius - 0.2, inner_radius + 0.2)
            x_pos = radius_variation * np.cos(angle + j * 0.4)
            y_pos = radius_variation * np.sin(angle + j * 0.4)
            img = np.squeeze(images[idx].transpose(1, 2, 0))
            ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    # Plot images for classes 5-9 (mid radius)
    for i, angle in enumerate(mid_angles):
        class_indices = np.where(true_labels == i + 5)[0]
        correctly_classified = class_indices[(predicted_labels[class_indices] == true_labels[class_indices])]
        incorrectly_classified = class_indices[(true_labels[class_indices] == num_classes)]

        # Select 5 correct samples for each class (MNIST)
        correctly_classified = correctly_classified[:5]
        incorrectly_classified = incorrectly_classified[:2]

        # Scatter points for correct classifications
        for j, idx in enumerate(correctly_classified):
            radius_variation = np.random.uniform(mid_radius - 0.2, mid_radius + 0.2)
            x_pos = radius_variation * np.cos(angle + j * 0.4)
            y_pos = radius_variation * np.sin(angle + j * 0.4)
            img = np.squeeze(images[idx].transpose(1, 2, 0))
            ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        # Scatter points for incorrect classifications (OOD classified as MNIST)
        for j, idx in enumerate(incorrectly_classified):
            radius_variation = np.random.uniform(mid_radius - 0.2, mid_radius + 0.2)
            x_pos = radius_variation * np.cos(angle + j * 0.4)
            y_pos = radius_variation * np.sin(angle + j * 0.4)
            img = np.squeeze(images[idx].transpose(1, 2, 0))
            ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    # Plot OOD images (outer radius)
    ood_indices = np.where(true_labels == num_classes)[0]
    correctly_classified_ood = ood_indices[np.where(predicted_labels[ood_indices] == num_classes)]
    incorrectly_classified_ood = np.where((true_labels != num_classes) & (predicted_labels == num_classes))[0]

    # Select 35 correct and 5 incorrect samples for OOD
    correctly_classified_ood = correctly_classified_ood[:35]
    incorrectly_classified_ood = incorrectly_classified_ood[:5]

    # Scatter points for correct OOD classifications
    for j, idx in enumerate(correctly_classified_ood):
        radius_variation = np.random.uniform(outer_radius - 0.2, outer_radius + 0.2)
        angle = np.random.uniform(0, 2 * np.pi)
        x_pos = radius_variation * np.cos(angle)
        y_pos = radius_variation * np.sin(angle)
        img = np.squeeze(images[idx].transpose(1, 2, 0))
        ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
        rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    # Scatter points for incorrectly classified MNIST as OOD
    for j, idx in enumerate(incorrectly_classified_ood):
        radius_variation = np.random.uniform(outer_radius - 0.2, outer_radius + 0.2)
        angle = np.random.uniform(0, 2 * np.pi)
        x_pos = radius_variation * np.cos(angle)
        y_pos = radius_variation * np.sin(angle)
        img = np.squeeze(images[idx].transpose(1, 2, 0))
        ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
        rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.axis('off')
    plt.show()


def plot_baseline_loss_and_accuracy_over_epochs(loss_history, accuracy_history):
    """
    Plot both the loss and accuracy over epochs.

    Parameters:
    - loss_history: List of loss values over the epochs.
    - accuracy_history: List of accuracy values over the epochs.
    """
    num_epochs = len(loss_history)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting the loss history
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, num_epochs + 1), loss_history, marker='o', color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True)

    # Create a second y-axis for the accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(range(1, num_epochs + 1), accuracy_history, marker='o', color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Add a title and show the plot
    fig.suptitle('Training Loss and Accuracy Over Epochs BaseLine model')
    fig.tight_layout()  # Adjust layout to make room for the title
    plt.show()


def plot_confusion_matrices(cm_list, titles, save_path=None, display=True):
    """
    Plots a list of confusion matrices with their respective titles using ConfusionMatrixDisplay.

    Parameters:
    - cm_list: List of confusion matrices (as numpy arrays or tensors).
    - titles: List of titles corresponding to each confusion matrix.
    - save_path: Optional. Path to save the plot image.
    - display: Optional. If True, displays the plot. If False, closes the plot after saving.
    """

    num_matrices = len(cm_list)
    cols = min(num_matrices, 3)  # Display up to 3 confusion matrices per row
    rows = (num_matrices + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Ensure axes is iterable even when there's only one matrix
    if num_matrices == 1:
        axes = [axes]

    for i, ax in enumerate(axes.flatten()):
        if i < num_matrices:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_list[i])
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(titles[i])
        else:
            ax.axis('off')  # Turn off unused subplots

    plt.tight_layout()

    # Save the plot to the specified file path if provided
    if save_path:
        # Extract the directory path
        dir_path = os.path.dirname(save_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path)

    # Display the plot
    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_rectangular_classification(model, data_loader, num_classes=10):
    """
    Evaluate the model and plot images in a rectangular layout with specific points for in-distribution classes
    and corners for OOD classifications.

    Parameters:
    - model: The trained model to evaluate.
    - data_loader: PyTorch DataLoader containing the test data.
    - num_classes: Number of in-distribution classes (default is 10 for MNIST).
    """
    show_correctly_classified_in_dist = 15
    show_misclassified_in_dist = 3
    show_correctly_classified_ood = 50
    show_misclassified_ood = 10

    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to hold true labels, predictions, and images
    all_preds = []
    all_labels = []
    images = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in data_loader:
            images.extend(inputs.cpu().numpy())
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    predicted_labels = np.array(all_preds)
    true_labels = np.array(all_labels)
    images = np.array(images)

    fig, ax = plt.subplots(figsize=(15, 15))
    fig.patch.set_facecolor('black')  # Set the figure background to black
    ax.set_facecolor('black')  # Set the axes background to black

    # Define positions for each class center
    class_centers = {
        0: (-7, 7),
        1: (-3, 7),
        2: (3, 7),
        3: (7, 7),
        4: (-7, 3),
        5: (7, 3),
        6: (-7, -3),
        7: (7, -3),
        8: (-3, -7),
        9: (3, -7)
    }

    # Define corners for OOD images
    ood_positions = [
        (-10, 10), (10, 10), (-10, -10), (10, -10)
    ]

    # Plot in-distribution images around class centers
    for i in range(num_classes):
        true_class_indices = np.where(true_labels == i)[0]
        predicted_as_class_indices = np.where(predicted_labels == i)[0]
        correctly_classified = np.intersect1d(true_class_indices, predicted_as_class_indices)

        # Incorrectly classified OOD as this class
        incorrectly_classified = np.intersect1d(np.where(true_labels == num_classes)[0], predicted_as_class_indices)

        # Select 15 correct samples and 3 incorrect samples
        correctly_classified = correctly_classified[:show_correctly_classified_in_dist]
        incorrectly_classified = incorrectly_classified[:show_misclassified_in_dist]

        x_center, y_center = class_centers[i]

        # Scatter points for correct classifications
        for j, idx in enumerate(correctly_classified):
            x_pos = np.random.uniform(x_center - 1, x_center + 1)
            y_pos = np.random.uniform(y_center - 1, y_center + 1)
            img = np.squeeze(images[idx].transpose(1, 2, 0))
            ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        # Scatter points for incorrectly classified OOD as this MNIST class
        for j, idx in enumerate(incorrectly_classified):
            x_pos = np.random.uniform(x_center - 1, x_center + 1)
            y_pos = np.random.uniform(y_center - 1, y_center + 1)
            img = np.squeeze(images[idx].transpose(1, 2, 0))
            ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    # Scatter points for OOD images at the corners
    ood_indices = np.where(true_labels == num_classes)[0]
    correctly_classified_ood = ood_indices[predicted_labels[ood_indices] == num_classes]
    incorrectly_classified_ood = np.where((true_labels != num_classes) & (predicted_labels == num_classes))[0]

    # Select 50 correct and 10 incorrect samples for OOD
    correctly_classified_ood = correctly_classified_ood[:show_correctly_classified_ood]
    incorrectly_classified_ood = incorrectly_classified_ood[:show_misclassified_ood]

    for j, idx in enumerate(correctly_classified_ood):
        corner = ood_positions[j % 4]
        x_pos = np.random.uniform(corner[0] - 1, corner[0] + 1)
        y_pos = np.random.uniform(corner[1] - 1, corner[1] + 1)
        img = np.squeeze(images[idx].transpose(1, 2, 0))
        # Ensure OOD images are always framed in red
        ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
        if true_labels[idx] == num_classes:
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='red', facecolor='none')
        else:
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    for j, idx in enumerate(incorrectly_classified_ood):
        corner = ood_positions[j % 4]
        x_pos = np.random.uniform(corner[0] - 1, corner[0] + 1)
        y_pos = np.random.uniform(corner[1] - 1, corner[1] + 1)
        img = np.squeeze(images[idx].transpose(1, 2, 0))
        # Ensure MNIST images (misclassified as OOD) are framed in blue
        ax.imshow(img, extent=[x_pos - 0.4, x_pos + 0.4, y_pos - 0.4, y_pos + 0.4], cmap='gray')
        if true_labels[idx] == num_classes:
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='red', facecolor='none')
        else:
            rect = Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.axis('off')
    plt.show()
