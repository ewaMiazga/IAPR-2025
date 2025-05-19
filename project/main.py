from loader import CocoDataset, UnlabeledImageFolder, VOCDataset, ChocolatePatchDataset, PatchTestDataset
from models.cnn import SimpleCNN
from models.mobile import LightFasterRCNNMobileNetV3
from helper import get_device, draw_boxes_on_image, save_patches
from trainer import Trainer
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import torch.nn as nn
import os 
import cv2
import shutil
import tqdm
import numpy as np
import json
import pandas as pd 
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import os
from collections import defaultdict

device = get_device()
print(f"Using device: {device}")

# Cell 2: Define paths and load annotation info
base_dir = os.path.dirname(os.path.abspath(__file__))
train_img_dir = os.path.join(base_dir, "dataset_project_iapr2025", "train_annotated")
ann_path = os.path.join(base_dir, "dataset_project_iapr2025", "train_annotated", "_annotations.coco.json")

# Load COCO annotations and exclude "objects" class
with open(ann_path, "r") as f:
    coco_json = json.load(f)

# ✅ Exclude 'objects' and assign label IDs starting from 1
categories = [cat for cat in coco_json["categories"] if cat["name"] != "objects"]
class_name_to_id = {cat["name"]: i + 1 for i, cat in enumerate(categories)}
num_classes = max(class_name_to_id.values()) + 1  # +1 for background class 0

# Display the class structure
print("Detected classes (excluding 'objects'):", list(class_name_to_id.keys()))
print("Total (with background):", num_classes)


def get_transform():
    return T.Compose([
        T.ToTensor(),  # Converts PIL image to tensor
    ])

# This collate_fn works for batched Faster R-CNN inputs
def collate_fn(batch):
    return tuple(zip(*batch))

# Load full dataset (with all classes including "objects")
full_dataset = CocoDataset(
    root=train_img_dir,
    annotation=ann_path,
    transform=get_transform()
)

# Split full dataset into 90% train / 10% validation
total_len = len(full_dataset)
train_len = int(0.9 * total_len)
val_len = total_len - train_len

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len])

### TODO: Add test_loader if needed
base_dir = os.path.dirname(os.path.abspath(__file__))
test_image_dir = os.path.join(base_dir, "dataset_project_iapr2025", "test")
unlabeled_dataset = UnlabeledImageFolder(test_image_dir)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False)
#test_loader = unlabeled_dataset

print(f"Loaded full dataset: {total_len} images -> {train_len} train / {val_len} val")

model = LightFasterRCNNMobileNetV3(num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
epochs = 1

trainer = Trainer(model=model,
                  model_name="MobileNetV3",
                      optimizer=optimizer,
                      num_epochs=epochs,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      device=device)

trainer.train()

avg_loss, overall_accuracy, overall_f1, per_class_f1, _ = trainer.evaluate()
print(f"Average Loss: {avg_loss}, Overall Accuracy: {overall_accuracy}%, Overall F1: {overall_f1}")


class UnlabeledImageFolder(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        self.transform = transform if transform else T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.image_paths[idx]
    
# Normalization map based on your CSV
NORMALIZED_LABEL_MAP = {
    "amandina": "Amandina",
    "arabia": "Arabia",
    "comtesse": "Comtesse",
    "creme brulee": "CrÃ¨me brulÃ©e",
    "crème brûlée": "CrÃ¨me brulÃ©e",
    "jelly black": "Jelly Black",
    "jelly milk": "Jelly Milk",
    "jelly white": "Jelly White",
    "noblesse": "Noblesse",
    "noir authentique": "Noir Authentique",
    "noir authentique": "Noir authentique",
    "passion au lait": "Passion au lait",  # ✅ match CSV
    "passion_aulait": "Passion au lait",   # catch potential typo
    "stracciatella": "Stracciatella",
    "tentation noir": "Tentation Noir",
    "tentation noir": "Tentation noir",
    "triangolo": "Triangolo"
}



def normalize_label(label):
    key = label.lower().replace('_', ' ').strip()
    return NORMALIZED_LABEL_MAP.get(key, key)

def predict_unlabeled_images_and_save(model, dataset, class_id_to_name, device, reference_df, output_csv="mobilenet_submission.csv", score_threshold=0.7):
    model.eval()
    results = []

    for i in range(len(dataset)):
        image, image_path = dataset[i]
        image_tensor = image.to(device).unsqueeze(0)

        with torch.no_grad():
            prediction = model(image_tensor)[0]

        scores = prediction['scores'].cpu()
        labels = prediction['labels'].cpu()

        class_counts = defaultdict(int)
        for score, label in zip(scores, labels):
            if score < score_threshold:
                continue
            raw_label = class_id_to_name.get(label.item(), f"ID {label.item()}")
            label_name = normalize_label(raw_label)
            class_counts[label_name] += 1

        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image_id = ''.join(filter(str.isdigit, image_id))

        row = {"id": image_id}
        for class_name in reference_df.columns[1:]:
            row[class_name] = class_counts.get(class_name, 0)

        results.append(row)

    df = pd.DataFrame(results)
    df = df[["id"] + reference_df.columns[1:].tolist()]
    df.to_csv(output_csv, index=False)
    print(f"✅ Submission saved to {output_csv}")

class_id_to_name = {v: k for k, v in class_name_to_id.items()}

# Load your fixed CSV (with 'Creme Brulee' instead of corrupted name)
base_dir = os.path.dirname(os.path.abspath(__file__))
reference_path = os.path.join(base_dir, "dataset_project_iapr2025", "final_submission.csv")
reference_df = pd.read_csv(reference_path)
# reference_df = pd.read_csv("dataset_project_iapr2025/final_submission.csv")

reference_df.columns = [
    "Crème brulée" if col == "Creme Brulee" else col for col in reference_df.columns
]

# Define dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
test_image_dir = os.path.join(base_dir, "dataset_project_iapr2025", "test")
# test_image_dir = "dataset_project_iapr2025/test"
unlabeled_dataset = UnlabeledImageFolder(test_image_dir)

# Invert class_name_to_id
class_id_to_name = {v: k for k, v in class_name_to_id.items()}

# Run prediction
predict_unlabeled_images_and_save(
    model,
    unlabeled_dataset,
    class_id_to_name,
    device,
    reference_df,
    output_csv="test.csv",
    score_threshold=0.7
)