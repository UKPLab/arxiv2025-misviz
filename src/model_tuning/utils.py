import os
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.transforms import v2 as transforms


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transform(augmentations):
    transform_list = []

    if "rotation" in augmentations:
        transform_list.append(transforms.RandomRotation(3))

    if "perspective" in augmentations:
        transform_list.append(transforms.RandomPerspective(0.1))

    return transform_list


class ClassifierHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feature_vector):
        return self.classifier(feature_vector)


def plot_confusion_matrix(y_true, y_pred, class_names, output_folder, name_suffix=""):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            f"{name_suffix}confusion_matrix.png",
        )
    )


def plot_losses(training_losses, validation_losses, output_path):
    _, ax = plt.subplots(figsize=(10, 6))
    x_coordinates = np.arange(len(training_losses))
    ax.plot(x_coordinates, training_losses, label="Training Loss")
    ax.plot(x_coordinates, validation_losses, label="Validation Loss")
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per Iteration", wrap=True)
    ax.legend()
    plt.savefig(
        os.path.join(
            output_path,
            "evaluation",
            f"losses.png",
        )
    )


def group_misclassifications(actual_labels, predicted_labels):
    if len(actual_labels) != len(predicted_labels):
        raise ValueError("The length of actual and predicted labels must be the same.")

    misclassifications = {}

    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual != predicted:
            if predicted not in misclassifications:
                misclassifications[predicted] = []
            misclassifications[predicted].append(actual)

    return misclassifications


def get_available_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Created by user Nawras on https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop_check(self, validation_loss):
        if (validation_loss + self.min_delta) < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (validation_loss + self.min_delta) > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def extract_split_and_indices(data_to_split):
    val_indices = []
    train_indices = []
    small_train_indices = []
    test_indices = []

    for i, entry in enumerate(data_to_split):
        if entry["split"] == "val":
            val_indices.append(i)
        elif "train" in entry["split"]:
            train_indices.append(i)
        elif entry["split"] == "test":
            test_indices.append(i)
        elif entry["split"] == "train small":
            small_train_indices.append(i)

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
        "train_small": small_train_indices,
    }
