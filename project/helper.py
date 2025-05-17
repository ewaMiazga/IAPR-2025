# Helper functions for file operations and data processing

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from tqdm import tqdm
import torch
from collections import defaultdict
import os
import csv
import os
import zipfile
import os
import zipfile
import cv2

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def get_device():
    """
    Get the appropriate device for tensor computations.

    Returns:
    - torch.device: Device (GPU, MPS, or CPU) available for tensor computations.
    """
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Check if Multi-Process Service (MPS) is available
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    # If neither CUDA nor MPS is available, use CPU
    else:
        return torch.device('cpu')
    
def convert_for_submission(data, output_file):
    """
    Convert the given dictionary into the desired CSV format and save it to a file.

    Parameters:
    data (dict): The input dictionary with IDs and lists of coin labels.
    output_file (str): The name of the output CSV file.
    """
    ...
    return ...

def get_confussion_matrix(y_pred_tensor, test_data, class_names):
    """
    Generate and plot a confusion matrix for the model predictions.
    """

    # 2. steup confusion instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=len(class_names), task="multilabel")
    confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)

    # 3.Plot a conf martrix
    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                    colorbar=True,
                                    show_normed=True,
                                    figsize=(10, 7),
                                    class_names=class_names)
    
    # 4. Save the confusion matrix plot
    plt.savefig("confusion_matrix.png")
    plt.show()

def unzip_to(zip_path: str, dest_dir: str) -> None:
    """
    Extracts all contents of the ZIP file at zip_path into the directory dest_dir.

    Args:
        zip_path: Path to the .zip archive.
        dest_dir: Directory to extract files into (will be created if it doesn't exist).
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Open and extract all
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(dest_dir)
def unzip_to(zip_path: str, dest_dir: str) -> None:
    """
    Extracts all contents of the ZIP file at zip_path into the directory dest_dir.

    Args:
        zip_path: Path to the .zip archive.
        dest_dir: Directory to extract files into (will be created if it doesn't exist).
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Open and extract all
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(dest_dir)


def compute_mean_std(loader):
    """
    Compute the mean and standard deviation of the dataset.
    Args:
        loader (DataLoader): DataLoader for the dataset.
    """
    n_channels = 3
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    total_images = 0

    for images in tqdm(loader):
        if images.ndim == 4:  # (B, C, H, W)
            total_images += images.size(0)
            for i in range(n_channels):
                mean[i] += images[:, i, :, :].mean()
                std[i] += images[:, i, :, :].std()
        else:
            print(f"Skipping batch with shape: {images.shape}")

    mean /= total_images
    std /= total_images
    return mean, std


def draw_boxes_on_image(image_path, boxes, labels, scores, label_map_inv, threshold=0.0):
    """
    Draw bounding boxes on the image and return the modified image.
    Args:
        image_path (str): Path to the image file.
        boxes (list): List of bounding boxes.
        labels (list): List of labels corresponding to the boxes.
        scores (list): List of scores corresponding to the boxes.
        label_map_inv (dict): Inverted label map for converting labels to class names.
        threshold (float): Score threshold for displaying boxes.
    """
    image = cv2.imread(image_path)
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
        class_name = label_map_inv.get(label.item(), str(label.item()))
        text = f"{class_name} {score:.2f}"

        # Draw box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def save_patches(image_path, boxes, labels, scores, label_map_inv, dest_dir, threshold=0.5):
    """
    Save patches of the image based on the bounding boxes, labels, and scores.
    Args:
        image_path (str): Path to the image file.
        boxes (list): List of bounding boxes.
        labels (list): List of labels corresponding to the boxes.
        scores (list): List of scores corresponding to the boxes.
        label_map_inv (dict): Inverted label map for converting labels to class names.
        dest_dir (str): Directory to save the patches.
        threshold (float): Score threshold for saving patches.
    """
    img = cv2.imread(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create subfolder
    subfolder = os.path.join(dest_dir, image_name)
    os.makedirs(subfolder, exist_ok=True)

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < threshold:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
        patch = img[y1:y2, x1:x2]
        label_name = label_map_inv.get(label.item(), str(label.item()))
        patch_filename = f"{i:03d}.jpg"
        cv2.imwrite(os.path.join(subfolder, patch_filename), patch)