import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import json
import xml.etree.ElementTree as ET
from PIL import Image

class CocoDataset(Dataset):
    """
    Custom dataset for COCO format annotations.
    Args:
        root (str): Path to the root directory containing images.
        annotation (str): Path to the COCO format annotation file.
        transform (callable, optional): Optional transform to be applied on a sample.
        has_annotations (bool): Whether the dataset has annotations or not.
    """
    def __init__(self, root, annotation=None, transform=None, has_annotations=True):
        self.root = root
        self.transform = transform if transform else T.ToTensor()
        self.has_annotations = has_annotations

        if self.has_annotations:
            with open(annotation, "r") as f:
                data = json.load(f)

            # Filter out category 'objects' and remap IDs starting from 1
            categories = [cat for cat in data["categories"] if cat["name"] != "objects"]
            self.class_name_to_id = {cat["name"]: i + 1 for i, cat in enumerate(categories)}
            self.old_to_new_id = {cat["id"]: self.class_name_to_id[cat["name"]] for cat in categories}

            self.id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

            self.image_annotations = {}
            for ann in data["annotations"]:
                if ann["category_id"] in self.old_to_new_id:
                    img_id = ann["image_id"]
                    if img_id not in self.image_annotations:
                        self.image_annotations[img_id] = []
                    self.image_annotations[img_id].append(ann)

            self.image_ids = list(self.image_annotations.keys())
        else:
            # No annotations: just list image files
            self.image_filenames = sorted([
                f for f in os.listdir(root)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

    def __len__(self):
        return len(self.image_ids) if self.has_annotations else len(self.image_filenames)

    def __getitem__(self, idx):
        if self.has_annotations:
            image_id = self.image_ids[idx]
            image_path = os.path.join(self.root, self.id_to_filename[image_id])
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)

            anns = self.image_annotations.get(image_id, [])
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(self.old_to_new_id[ann["category_id"]])

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4)),
                "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
            }

            return image_tensor, target
        else:
            image_path = os.path.join(self.root, self.image_filenames[idx])
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor, self.image_filenames[idx]  # return image name to match outputs
        
    
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


class VOCDataset(Dataset):
    def __init__(self, root, transform=None, label_map=None):
        self.root = root
        self.images = [f for f in os.listdir(root) if f.endswith(".jpg")]
        self.transform = transform if transform else T.ToTensor()
        
        # Class name to index mapping
        self.label_map = label_map

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_filename = self.images[idx]
        xml_filename = image_filename.replace(".jpg", ".xml")

        image_path = os.path.join(self.root, image_filename)
        xml_path = os.path.join(self.root, xml_filename)

        image = Image.open(image_path).convert("RGB")
        boxes = []
        labels = []

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name").text
            if self.label_map:
                label = self.label_map[name]
            else:
                label = name  # return string if no mapping
            
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transform:
            image = self.transform(image)

        return image, target
    

    

### Classification
class ChocolatePatchDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['name'] + '.jpg')
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row['class_idx'], dtype=torch.long)
        return image, label
    
    from torch.utils.data import Dataset


class PatchTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to root patch folder (e.g., results/patches)
        """
        self.root_dir = root_dir
        self.transform = transform if transform else T.ToTensor()

        # Gather all patch paths + image IDs
        self.samples = []
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for file in sorted(os.listdir(folder_path)):
                if file.lower().endswith(".jpg"):
                    self.samples.append({
                        "img_path": os.path.join(folder_path, file),
                        "parent_id": folder,
                        "patch_name": file
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample["parent_id"], sample["patch_name"]