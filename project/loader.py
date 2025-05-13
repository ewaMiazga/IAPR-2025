import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CustomIAPRDataloader:
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.train_csv = pd.read_csv(os.path.join(base_dir, 'train.csv'))

        self.train_dataset = self._create_train_dataset()
        self.test_dataset = self._create_test_dataset()
        self.reference_dataset = self._create_reference_dataset()
        self.class_number = len(self.train_csv.columns) - 1  # Assuming first column is 'id'
        self.class_names = self.train_csv.columns[1:].tolist()  # Assuming first column is 'id'
    

    def _create_train_dataset(self):
        return TrainDataset(
            image_dir=os.path.join(self.base_dir, 'train'),
            dataframe=self.train_csv,
            transform=self.transform
        )

    def _create_test_dataset(self):
        return TestDataset(
            image_dir=os.path.join(self.base_dir, 'test'),
            transform=self.transform
        )

    def _create_reference_dataset(self):
        return ReferenceDataset(
            image_dir=os.path.join(self.base_dir, 'references'),
            transform=self.transform
        )

        
class TrainDataset(Dataset):
    def __init__(self, image_dir, dataframe, transform):
        self.image_dir = image_dir
        self.df = dataframe
        self.transform = transform
        self.df['id'] = self.df['id'].astype(str).str.zfill(7)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = f"L{row['id']}.JPG"
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        labels = torch.tensor(row.iloc[1:].values.astype(int), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels
    
class TestDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    

class ReferenceDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]
    

class ImageOnlyDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.copy()
        self.df['id'] = self.df['id'].astype(str).str.zfill(7)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = f"L{self.df.iloc[idx]['id']}.JPG"
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image