"""
Phase 8 - Contribution 3: Universal Domain Gap Validation (DFDC)
================================================================
Copy-paste this entire script into a Kaggle cell and run it.
Make sure you have added these two inputs:
  - deepfake-model-weights (your .pth files)
  - dfdc-faces-of-the-train-sample (the DFDC face dataset)
"""

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# DFDC PATHS (using validation set for zero-shot evaluation)
# ==============================================================================
DFDC_REAL_DIR = "/kaggle/input/datasets/itamargr/dfdc-faces-of-the-train-sample/validation/real"
DFDC_FAKE_DIR = "/kaggle/input/datasets/itamargr/dfdc-faces-of-the-train-sample/validation/fake"

# ==============================================================================
# DATASET CLASS
# ==============================================================================
class SimpleDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================================================================
# LOAD MODEL
# ==============================================================================
def load_model(weights_path):
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Loaded: {os.path.basename(weights_path)}")
    return model

# ==============================================================================
# EVALUATE
# ==============================================================================
def evaluate_model(model, dataloader, name):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().flatten())
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\n  {name}:")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    AUC-ROC:  {auc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    return acc, auc, all_labels, all_preds

# ==============================================================================
# MAIN
# ==============================================================================
print("=" * 60)
print("PHASE 8: UNIVERSAL DOMAIN GAP VALIDATION (DFDC)")
print("=" * 60)

# Collect image paths
real_images = sorted(glob.glob(os.path.join(DFDC_REAL_DIR, "*.png")))
if not real_images:
    real_images = sorted(glob.glob(os.path.join(DFDC_REAL_DIR, "**/*.png"), recursive=True))
fake_images = sorted(glob.glob(os.path.join(DFDC_FAKE_DIR, "*.png")))
if not fake_images:
    fake_images = sorted(glob.glob(os.path.join(DFDC_FAKE_DIR, "**/*.png"), recursive=True))

print(f"\nDFDC validation set: {len(real_images)} real + {len(fake_images)} fake")

# Cap at 5000 per class
MAX_PER_CLASS = 5000
if len(real_images) > MAX_PER_CLASS:
    real_images = real_images[:MAX_PER_CLASS]
    print(f"  Capped real to {MAX_PER_CLASS}")
if len(fake_images) > MAX_PER_CLASS:
    fake_images = fake_images[:MAX_PER_CLASS]
    print(f"  Capped fake to {MAX_PER_CLASS}")

all_paths = real_images + fake_images
all_labels = [0]*len(real_images) + [1]*len(fake_images)

dataset = SimpleDataset(all_paths, all_labels, test_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Find model weights
exp1_path = "/kaggle/input/datasets/simmi90/deepfake-model-weights/Exp1_Baseline_best.pth"
exp2_path = "/kaggle/input/datasets/simmi90/deepfake-model-weights/Exp2_Robust_best.pth"

# Evaluate both models
print("\n" + "=" * 60)
print("ZERO-SHOT CROSS-DATASET EVALUATION ON DFDC")
print("=" * 60)

model1 = load_model(exp1_path)
acc1, auc1, labels1, preds1 = evaluate_model(model1, dataloader, "Exp1 Baseline on DFDC")
del model1
torch.cuda.empty_cache()

model2 = load_model(exp2_path)
acc2, auc2, labels2, preds2 = evaluate_model(model2, dataloader, "Exp2 Robust on DFDC")
del model2
torch.cuda.empty_cache()

# Results
print("\n" + "=" * 60)
print("DFDC RESULTS SUMMARY")
print("=" * 60)
print(f"Exp1 Baseline on DFDC:  Acc={acc1:.4f}  AUC={auc1:.4f}")
print(f"Exp2 Robust on DFDC:    Acc={acc2:.4f}  AUC={auc2:.4f}")
print("=" * 60)

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm1 = confusion_matrix(labels1, preds1)
cm2 = confusion_matrix(labels2, preds2)

sns.heatmap(cm1, annot=True, fmt='d', cmap='Reds', ax=axes[0],
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
axes[0].set_title('Exp1: Baseline on DFDC', fontsize=13)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
axes[1].set_title('Exp2: Robust on DFDC', fontsize=13)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('/kaggle/working/dfdc_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: dfdc_confusion_matrices.png")

# Tri-Dataset Comparison
print("\n" + "=" * 70)
print("TRI-DATASET COMPARISON (FF++ vs Celeb-DF vs DFDC)")
print("=" * 70)
print(f"{'Dataset':<25} {'Exp1 Baseline':<20} {'Exp2 Robust':<20}")
print(f"{'':.<25} {'Acc / AUC':.<20} {'Acc / AUC':.<20}")
print("-" * 70)
print(f"{'FF++ (c23→c40)':<25} {'92.80 / 99.38':<20} {'99.05 / 99.97':<20}")
print(f"{'Celeb-DF (zero-shot)':<25} {'42.39 / 82.85':<20} {'39.56 / 79.91':<20}")
print(f"{'DFDC (zero-shot)':<25} {f'{acc1*100:.2f} / {auc1*100:.2f}':<20} {f'{acc2*100:.2f} / {auc2*100:.2f}':<20}")
print("=" * 70)

# Tri-Dataset Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

datasets_list = ['FF++\n(c23→c40)', 'Celeb-DF\n(zero-shot)', 'DFDC\n(zero-shot)']
baseline_accs = [92.80, 42.39, acc1*100]
robust_accs = [99.05, 39.56, acc2*100]
baseline_aucs = [99.38, 82.85, auc1*100]
robust_aucs = [99.97, 79.91, auc2*100]

x = np.arange(len(datasets_list))
width = 0.35

bars1 = axes[0].bar(x - width/2, baseline_accs, width, label='Exp1: Baseline', color='#e74c3c')
bars2 = axes[0].bar(x + width/2, robust_accs, width, label='Exp2: Robust', color='#2ecc71')
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Accuracy: Tri-Dataset Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(datasets_list)
axes[0].legend()
axes[0].set_ylim(0, 115)
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

bars3 = axes[1].bar(x - width/2, baseline_aucs, width, label='Exp1: Baseline', color='#c0392b')
bars4 = axes[1].bar(x + width/2, robust_aucs, width, label='Exp2: Robust', color='#27ae60')
axes[1].set_ylabel('AUC-ROC (%)', fontsize=12)
axes[1].set_title('AUC-ROC: Tri-Dataset Comparison', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(datasets_list)
axes[1].legend()
axes[1].set_ylim(0, 115)
for bar in bars3:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
for bar in bars4:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/tri_dataset_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: tri_dataset_comparison.png")

print("\nAll DFDC results saved to /kaggle/working/")
print("PHASE 8 CONTRIBUTION 3 COMPLETE!")
