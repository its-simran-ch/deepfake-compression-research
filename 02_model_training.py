"""
Model Training Script for Deepfake Detection
=============================================
Runs two experiments using EfficientNet-B4 on extracted face data.

INSTRUCTIONS:
1. Create a NEW Kaggle Notebook called "Deepfake Model Training"
2. Turn on GPU (T4 x2 or P100)
3. Turn on Internet
4. Click "+ Add Input" -> "Your Work" -> "Datasets" -> select "ff-extracted-faces"
5. Run Cell 1: !pip install timm
6. Run Cell 2: Paste this entire script
"""

import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path to the extracted faces dataset on Kaggle
# The dataset was uploaded as zip files (c23.zip & c40.zip), Kaggle auto-extracts them
DATA_DIR = "/kaggle/input/datasets/simranch77/ff-extracted-faces"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# DATASET & DATALOADERS
# ==============================================================================
class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # If image is corrupted, return a blank image
            image = Image.new("RGB", (224, 224))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

def get_image_paths_and_labels(data_dir, compression):
    """Helper to get all image paths for a specific compression level"""
    real_paths = sorted(glob.glob(os.path.join(data_dir, compression, "real", "*", "*.jpg")))
    fake_paths = sorted(glob.glob(os.path.join(data_dir, compression, "fake", "*", "*.jpg")))
    
    print(f"  {compression}/real: {len(real_paths)} images")
    print(f"  {compression}/fake: {len(fake_paths)} images")
    
    paths = real_paths + fake_paths
    labels = [0]*len(real_paths) + [1]*len(fake_paths)
    return paths, labels

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================================================================
# MODEL
# ==============================================================================
def create_model():
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1)
    )
    return model.to(device)

# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================
def train_model(model, train_loader, val_loader, experiment_name):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    
    print(f"\n{'='*60}")
    print(f"Starting Training: {experiment_name}")
    print(f"{'='*60}")
    
    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        val_auc = roc_auc_score(all_labels, all_probs)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"/kaggle/working/{experiment_name}_best.pth")
            print("  --> Saved new best model!")
    
    # Save history as CSV
    with open(f"/kaggle/working/{experiment_name}_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_auc"])
        for i in range(EPOCHS):
            writer.writerow([i+1, history['train_loss'][i], history['val_loss'][i], history['val_acc'][i], history['val_auc'][i]])
    
    return history, all_labels, all_preds, all_probs

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================
def plot_comparison(h1, h2):
    """Plot training curves comparing both experiments"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, EPOCHS+1)
    
    # Loss
    axes[0].plot(epochs, h1['val_loss'], 'r-o', label='Exp1: Baseline (c23→c40)')
    axes[0].plot(epochs, h2['val_loss'], 'g-o', label='Exp2: Robust (mixed→c40)')
    axes[0].set_title('Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, h1['val_acc'], 'r-o', label='Exp1: Baseline')
    axes[1].plot(epochs, h2['val_acc'], 'g-o', label='Exp2: Robust')
    axes[1].set_title('Validation Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    # AUC
    axes[2].plot(epochs, h1['val_auc'], 'r-o', label='Exp1: Baseline')
    axes[2].plot(epochs, h2['val_auc'], 'g-o', label='Exp2: Robust')
    axes[2].set_title('Validation AUC-ROC', fontsize=14)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('AUC')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/comparison_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: comparison_curves.png")

def plot_confusion_matrices(labels1, preds1, labels2, preds2):
    """Plot confusion matrices side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm1 = confusion_matrix(labels1, preds1)
    cm2 = confusion_matrix(labels2, preds2)
    
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    axes[0].set_title('Exp1: Baseline (c23→c40)', fontsize=13)
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    axes[1].set_title('Exp2: Robust (mixed→c40)', fontsize=13)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: confusion_matrices.png")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("Loading data paths...")
    c23_paths, c23_labels = get_image_paths_and_labels(DATA_DIR, "c23")
    c40_paths, c40_labels = get_image_paths_and_labels(DATA_DIR, "c40")
    
    if len(c23_paths) == 0 or len(c40_paths) == 0:
        print("\nERROR: No images found! Trying alternate path structure...")
        # Try checking what's inside the dataset
        for root, dirs, files in os.walk(DATA_DIR):
            level = root.replace(DATA_DIR, "").count(os.sep)
            if level <= 3:
                print(f"{'  '*level}{os.path.basename(root)}/  ({len(files)} files)")
        return
    
    # Shuffle data consistently
    c23_combined = list(zip(c23_paths, c23_labels))
    c40_combined = list(zip(c40_paths, c40_labels))
    random.shuffle(c23_combined)
    random.shuffle(c40_combined)
    c23_paths, c23_labels = zip(*c23_combined)
    c40_paths, c40_labels = zip(*c40_combined)
    c23_paths, c23_labels = list(c23_paths), list(c23_labels)
    c40_paths, c40_labels = list(c40_paths), list(c40_labels)
    
    # 80-20 split
    split = int(len(c23_paths) * 0.8)
    
    # ==========================================
    # EXPERIMENT 1: Baseline (Train c23, Test c40)
    # ==========================================
    print("\n>>> EXPERIMENT 1: BASELINE (Train on c23, Test on c40) <<<")
    train_paths_1 = c23_paths[:split]
    train_labels_1 = c23_labels[:split]
    val_paths_1 = c40_paths[split:]
    val_labels_1 = c40_labels[split:]
    
    print(f"  Train: {len(train_paths_1)} images (c23 only)")
    print(f"  Val:   {len(val_paths_1)} images (c40 only)")
    
    train_ds1 = DeepfakeDataset(train_paths_1, train_labels_1, train_transforms)
    val_ds1 = DeepfakeDataset(val_paths_1, val_labels_1, val_transforms)
    train_loader1 = DataLoader(train_ds1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader1 = DataLoader(val_ds1, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model1 = create_model()
    h1, labels1, preds1, probs1 = train_model(model1, train_loader1, val_loader1, "Exp1_Baseline")
    
    # ==========================================
    # EXPERIMENT 2: Robust (Train mixed c23+c40, Test c40)
    # ==========================================
    print("\n>>> EXPERIMENT 2: ROBUST (Train on mixed c23+c40, Test on c40) <<<")
    half = split // 2
    train_paths_2 = c23_paths[:half] + c40_paths[:half]
    train_labels_2 = c23_labels[:half] + c40_labels[:half]
    val_paths_2 = c40_paths[split:]
    val_labels_2 = c40_labels[split:]
    
    # Shuffle the mixed training data
    combined = list(zip(train_paths_2, train_labels_2))
    random.shuffle(combined)
    train_paths_2, train_labels_2 = zip(*combined)
    train_paths_2, train_labels_2 = list(train_paths_2), list(train_labels_2)
    
    print(f"  Train: {len(train_paths_2)} images (mixed c23+c40)")
    print(f"  Val:   {len(val_paths_2)} images (c40 only)")
    
    train_ds2 = DeepfakeDataset(train_paths_2, train_labels_2, train_transforms)
    val_ds2 = DeepfakeDataset(val_paths_2, val_labels_2, val_transforms)
    train_loader2 = DataLoader(train_ds2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader2 = DataLoader(val_ds2, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model2 = create_model()
    h2, labels2, preds2, probs2 = train_model(model2, train_loader2, val_loader2, "Exp2_Robust")
    
    # ==========================================
    # RESULTS & PLOTS
    # ==========================================
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    print(f"Experiment 1 (Baseline c23→c40):  Acc={h1['val_acc'][-1]:.4f}  AUC={h1['val_auc'][-1]:.4f}")
    print(f"Experiment 2 (Robust mixed→c40):  Acc={h2['val_acc'][-1]:.4f}  AUC={h2['val_auc'][-1]:.4f}")
    print("="*60)
    
    plot_comparison(h1, h2)
    plot_confusion_matrices(labels1, preds1, labels2, preds2)
    
    print("\nAll results saved to /kaggle/working/")
    print("Files: Exp1_Baseline_history.csv, Exp2_Robust_history.csv, comparison_curves.png, confusion_matrices.png")

if __name__ == "__main__":
    main()
