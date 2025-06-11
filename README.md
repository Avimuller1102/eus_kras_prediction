# EUS KRAS Prediction Project

This repository implements a deep learning pipeline to predict KRAS mutation status in pancreatic adenocarcinoma from endoscopic ultrasound (EUS) images.
By Avinoam Aharon Muller 

This project aims to develop a deep learning model to predict KRAS mutation status (and other key mutations) in pancreatic cancer using Endoscopic Ultrasound (EUS) images. While there are existing methods for tumor detection and classification in EUS, none have focused on predicting mutations like KRAS purely from imaging data. KRAS mutations are crucial for determining prognosis and treatment options, and this non-invasive approach could provide valuable insights without the need for additional biopsies, potentially speeding up decision-making and optimizing treatment plans for patients. The innovation lies in leveraging EUS images to predict molecular mutations, a concept not yet explored in current research.


avimuller1102@gmail.com

## Directory Structure
eus_kras_project/
├── config.yaml # configuration file
├── requirements.txt # Python dependencies
├── datasets/
│ └── dataset.py # Dataset class and transforms
├── models/
│ └── model.py # CNN model definition
├── utils.py # utility functions (metrics, EarlyStopping)
├── train.py # training script
├── evaluate.py # evaluation and Grad-CAM visualization
└── README.md # this file


## Setup
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare your data:
   - Place EUS images in `data/images/`.
   - Create `data/annotations.csv` with columns: `patient_id`, `image_path` (relative to `data/images/`), `kras_status` (0 or 1).

## Configuration
Edit `config.yaml` to set:
- Dataset paths (`annotations_csv`, `image_dir`).
- Model hyperparameters (`backbone`, `lr`, etc.).
- Training parameters (`epochs`, `batch_size`).

## Training
Run:
```bash
python train.py
```
Model checkpoints will be saved to `checkpoints/best_model.pth`.

## Evaluation
Run:
```bash
python evaluate.py
```
This will compute test metrics and save Grad-CAM visualizations to `results/`.
```
