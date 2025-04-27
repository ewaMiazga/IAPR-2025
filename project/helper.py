# Helper functions for file operations and data processing

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from tqdm import tqdm
import torch
from collections import defaultdict
import os
import csv

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

