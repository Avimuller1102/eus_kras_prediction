import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import cv2
from datasets.dataset import EUSDataset, get_val_transforms
from models.model import KRASClassifier
from utils import compute_metrics
from tqdm import tqdm

def generate_gradcam(model, image, device, target_class):
    model.eval()
    image = image.unsqueeze(0).to(device)
    features = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal features
        features = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    # Hook the last convolutional layer
    last_conv = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No convolutional layer found in backbone.")

    handle_forward = last_conv.register_forward_hook(forward_hook)
    handle_backward = last_conv.register_backward_hook(backward_hook)

    outputs = model(image)
    score = outputs[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    features = features[0]
    for i in range(features.shape[0]):
        features[i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(features, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (image.size(3), image.size(2)))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    handle_forward.remove()
    handle_backward.remove()
    return heatmap

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

# Create required directories
os.makedirs(config['evaluation']['output_dir'], exist_ok=True)

# Prepare test dataset and loader
test_dataset = EUSDataset(
    annotations_csv=config['data']['annotations_csv'],
    image_dir=config['data']['image_dir'],
    input_size=config['data']['input_size'],
    transforms=get_val_transforms(config['data']['input_size'])
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=False,
    num_workers=config['data']['num_workers'],
    pin_memory=True
)

# Initialize and load model
model = KRASClassifier(
    backbone=config['model']['backbone'],
    pretrained=False,
    num_classes=config['model']['num_classes']
).to(device)
model.load_state_dict(torch.load(os.path.join(config['training']['output_dir'], 'best_model.pth'), map_location=device))

# Evaluate on test set
all_labels = []
all_probs = []
model.eval()
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating on Test"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
print("Test Metrics:", metrics)

# Grad-CAM visualization (for first N samples)
if config['evaluation']['gradcam']:
    for idx in range(min(10, len(test_dataset))):
        image, label = test_dataset[idx]
        prob = torch.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0, 1].item()
        pred_class = 1 if prob >= 0.5 else 0
        heatmap = generate_gradcam(model, image, device, target_class=pred_class)
        # Overlay heatmap on original grayscale image
        orig = (image.squeeze(0).numpy() * 255).astype(np.uint8)
        orig_color = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(orig_color, 0.6, heatmap, 0.4, 0)
        # Save
        out_path = os.path.join(config['evaluation']['output_dir'], f"gradcam_{idx}_label{label}_pred{pred_class}.png")
        cv2.imwrite(out_path, overlay)
    print("Gradient CAMs saved.")
