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