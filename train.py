import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.dataset import EUSDataset, get_train_transforms, get_val_transforms
from models.model import KRASClassifier
from utils import compute_metrics, EarlyStopping
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

# Create required directories
os.makedirs(config['training']['output_dir'], exist_ok=True)
os.makedirs('data', exist_ok=True)

# Prepare datasets and dataloaders
train_df = pd.read_csv(config['data']['annotations_csv'])

# Split train/val
train_data, val_data = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df['kras_status'],
    random_state=42
)
train_csv = 'data/train_annotations.csv'
val_csv = 'data/val_annotations.csv'
train_data.to_csv(train_csv, index=False)
val_data.to_csv(val_csv, index=False)

train_dataset = EUSDataset(
    annotations_csv=train_csv,
    image_dir=config['data']['image_dir'],
    input_size=config['data']['input_size'],
    transforms=get_train_transforms(config['data']['input_size'])
)
val_dataset = EUSDataset(
    annotations_csv=val_csv,
    image_dir=config['data']['image_dir'],
    input_size=config['data']['input_size'],
    transforms=get_val_transforms(config['data']['input_size'])
)

dloader_train = DataLoader(
    train_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True,
    num_workers=config['data']['num_workers'],
    pin_memory=True
)
dloader_val = DataLoader(
    val_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=False,
    num_workers=config['data']['num_workers'],
    pin_memory=True
)

# Initialize model
model = KRASClassifier(
    backbone=config['model']['backbone'],
    pretrained=config['model']['pretrained'],
    num_classes=config['model']['num_classes']
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=config['model']['lr'],
    weight_decay=config['model']['weight_decay']
)

# Scheduler
if config['model']['scheduler'] == 'ReduceLROnPlateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['model']['scheduler_params']['mode'],
        factor=config['model']['scheduler_params']['factor'],
        patience=config['model']['scheduler_params']['patience']
    )
else:
    scheduler = None

# Early stopping
early_stopper = EarlyStopping(
    patience=config['training']['early_stopping_patience'],
    verbose=True
)

for epoch in range(config['training']['epochs']):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dloader_train, desc=f"Training Epoch {epoch+1}"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_train_loss = running_loss / len(dloader_train.dataset)

    model.eval()
    val_running_loss = 0.0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(dloader_val, desc=f"Validation Epoch {epoch+1}"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    epoch_val_loss = val_running_loss / len(dloader_val.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    print(f"Epoch {epoch+1}/{config['training']['epochs']}: Train Loss: {epoch_train_loss:.4f}, "
          f"Val Loss: {epoch_val_loss:.4f}, Val AUC: {metrics['auc']:.4f}")

    # Scheduler step
    if scheduler:
        scheduler.step(epoch_val_loss)

    # Early stopping and saving model
    ckpt_path = os.path.join(config['training']['output_dir'], "best_model.pth")
    early_stopper(epoch_val_loss, model, ckpt_path)
    if early_stopper.early_stop:
        print("Early stopping triggered. Stopping training.")
        break

print("Training complete.")
