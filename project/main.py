# Torch and torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Custom modules
from src.loader import CocoDataset, UnlabeledImageFolder
from models.mobile import LightFasterRCNNMobileNetV3
from src.trainer import Trainer
from src.helper import get_device, predict_unlabeled_images_and_save

# Other useful libraries
import os
import json
import pandas as pd
import numpy as np
import random

# For the reproducibility of results
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

# Set to True to train the model, False to evaluate
IS_TRAINING = False 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = "dataset_project_iapr2025"
TEST_DIR = test_image_dir = os.path.join(BASE_DIR, DATASET_DIR, "test")

# optimaal, but does not ensure reproducibility
#device = get_device()

# Keep this line for consistent results
device = "cpu"
print(f"Using device: {device}")



def get_transform():
    return T.Compose([
        T.ToTensor(),  # Converts PIL image to tensor
    ])

# This collate_fn works for batched Faster R-CNN inputs
def collate_fn(batch):
    return tuple(zip(*batch))

if IS_TRAINING:  
    # Cell 2: Define paths and load annotation info
    train_img_dir = os.path.join(BASE_DIR, DATASET_DIR, "train_annotated")
    ann_path = os.path.join(BASE_DIR, DATASET_DIR, "train_annotated", "_annotations.coco.json")

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
    unlabeled_dataset = UnlabeledImageFolder(TEST_DIR)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False)

    print(f"Loaded full dataset: {total_len} images -> {train_len} train / {val_len} val")


    model = LightFasterRCNNMobileNetV3(num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    epochs = 150
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


          
else: 
    num_classes = 14
    
    unlabeled_dataset = UnlabeledImageFolder(TEST_DIR)

    # Create data loaders
    train_loader = None
    val_loader = None
    test_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False)

    model = LightFasterRCNNMobileNetV3(num_classes=num_classes)
    epochs = 150
    checkpoint_path = os.path.join(BASE_DIR,"checkpoints", "MobilenetV3.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    trainer = Trainer(model=model,
                    model_name="MobileNetV3",
                        optimizer=optimizer,
                        num_epochs=epochs,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        device=device)

# Load your fixed CSV (with 'Creme Brulee' instead of corrupted name)
reference_path = os.path.join(BASE_DIR, "sample_csv", "csv_file_template.csv")
reference_df = pd.read_csv(reference_path)

# Invert class_name_to_id
class_name_to_id = {'Amandina': 1, 'Arabia': 2, 'Comtesse': 3, 'Creme_brulee': 4, 'Jelly_Black': 5, 'Jelly_Milk': 6, 'Jelly_White': 7, 'Noblesse': 8, 'Noir_authentique': 9, 'Passion_au_lait': 10, 'Stracciatella': 11, 'Tentation_noir': 12, 'Triangolo': 13}
class_id_to_name = {v: k for k, v in class_name_to_id.items()}

print("Saving predictions to csv_output/submission.csv")
print("It may take a while...")

# Run prediction
predict_unlabeled_images_and_save(
    model,
    unlabeled_dataset,
    class_id_to_name,
    device,
    reference_df,
    output_csv="csv_output/submission.csv",
    score_threshold=0.7
)