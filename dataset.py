import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EUSDataset(Dataset):
    def __init__(self, annotations_csv, image_dir, input_size, transforms=None):
        self.df = pd.read_csv(annotations_csv)
        self.image_dir = image_dir
        self.input_size = input_size
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_path'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        # Resize to square
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        label = int(row['kras_status'])  # 0 or 1
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        else:
            # convert to tensor
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        return image, label


def get_train_transforms(input_size):
    return A.Compose([
        A.Resize(input_size, input_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(max_holes=1, max_height=int(0.1 * input_size), max_width=int(0.1 * input_size), p=0.3),
        ToTensorV2(),
    ])


def get_val_transforms(input_size):
    return A.Compose([
        A.Resize(input_size, input_size, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(),
    ])
