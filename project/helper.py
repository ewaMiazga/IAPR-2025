# Helper functions for file operations and data processing

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from tqdm import tqdm
import torch
from collections import defaultdict
import os
import zipfile
import os
import zipfile
import cv2
import shutil
import json
import pandas as pd
from torchvision import transforms

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

## TODO:   
def convert_for_submission(data, output_file):
    """
    Convert the given dictionary into the desired CSV format and save it to a file.

    Parameters:
    data (dict): The input dictionary with IDs and lists of coin labels.
    output_file (str): The name of the output CSV file.
    """
    ...
    return ...


## not used
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


### not used
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

def extract_patch(image_path, bbox, size=1400):
    """
    Extract a patch from the image based on the bounding box coordinates.
    Args:
        image_path (str): Path to the image file.
        bbox (list): List of bounding box coordinates [x, y, width, height].
        size (int): Size of the patch to extract.
    """
    x, y, w, h = [int(coord) for coord in bbox]
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    patch = img[y:y+h, x:x+w]
    return patch

def patches_from_coco(source, dest):
    """
    If test_mode is True, extracts full-image patches from each image in the folder.
    Otherwise, uses COCO annotations to extract object patches with labels.
    """
    patches_dir = os.path.join(dest, "patches")

    if os.path.exists(patches_dir):
        shutil.rmtree(patches_dir)
    os.makedirs(patches_dir)

    annotations_file = os.path.join(source, "_annotations.coco.json")
    images_dir = source
    patches_dir = os.path.join(dest, "patches")

    if os.path.exists(patches_dir):
        shutil.rmtree(patches_dir)
    os.makedirs(patches_dir)

    data = json.load(open(annotations_file, "r"))

    id_to_label = {e["id"]: e["name"] for e in data["categories"]}
    id_to_images = {e["id"]: e["file_name"] for e in data["images"]}
    annotations = data["annotations"]

    df_labels = pd.DataFrame(columns=["name", "label", "image", "bbox"])

    for i, annotation in tqdm(enumerate(annotations), total=len(annotations)):
        image_id = annotation["image_id"]
        label_id = annotation["category_id"]
        bbox = annotation["bbox"]
        label = id_to_label[label_id]
        image_file = id_to_images[image_id]
        image_path = os.path.join(images_dir, image_file)

        try:
            patch = extract_patch(image_path, bbox)
            idx = str(i).zfill(3)
            patch_path = os.path.join(patches_dir, f"{idx}.jpg")
            cv2.imwrite(patch_path, patch)
            df_labels.loc[i] = [idx, label, image_file, bbox]
        except Exception as e:
            print(f"⚠️ Skipped patch {i} from image {image_file}: {e}")

    df_labels.to_csv(os.path.join(patches_dir, "labels.csv"), index=False)
    print(f"✅ Saved {len(df_labels)} patches and labels to {patches_dir}")

def patches_to_ImageFolder(src, dest):
    # Load the CSV with patch labels
    labels_csv = os.path.join(src, "labels.csv")
    df = pd.read_csv(labels_csv)

    # Clear and recreate the destination folder
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)

    # Create subfolders and copy images
    for label in tqdm(df["label"].unique(), desc="Creating folders"):
        label_dir = os.path.join(dest, label)
        os.makedirs(label_dir, exist_ok=True)

        for _, row in df[df["label"] == label].iterrows():
            patch_name = f"{str(row['name']).zfill(3)}.jpg"
            src_file = os.path.join(src, patch_name)
            dest_file = os.path.join(label_dir, patch_name)

            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)
            else:
                print(f"⚠️ Missing file: {src_file}")

def get_transform():
    return transforms.Compose([
        transforms.Resize((400, 600)),  # Resize to 1400x1400
        transforms.ToTensor(),  # Converts PIL image or ndarray to tensor
    ])

def collate_fn(batch):
    images, targets = zip(*batch)
    converted_targets = []
    for target in targets:
        boxes = torch.as_tensor([obj['bbox'] for obj in target], dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]  # Convert [x,y,w,h] to [x1,y1,x2,y2]
        labels = torch.as_tensor([obj['category_id'] for obj in target], dtype=torch.int64)
        converted_targets.append({'boxes': boxes, 'labels': labels})
    return list(images), converted_targets