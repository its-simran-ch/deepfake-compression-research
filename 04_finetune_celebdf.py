"""
Experiment 3: Fine-Tuning on Celeb-DF (Domain Adaptation)
=========================================================
Fine-tunes the FF++ robust model on Celeb-DF data for improved
cross-dataset generalization. Run this in a NEW CELL in the
same "Cross-Dataset Testing" notebook (faces are already extracted).
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
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5  # Lower LR for fine-tuning
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# LOAD CELEB-DF EXTRACTED FACES (already extracted in previous cell)
# ==============================================================================
CELEBDF_FACES = "/kaggle/working/celebdf_faces"

real_images = sorted(glob.glob(os.path.join(CELEBDF_FACES, "real", "*", "*.jpg")))
fake_images = sorted(glob.glob(os.path.join(CELEBDF_FACES, "fake", "*", "*.jpg")))

print(f"Celeb-DF faces: {len(real_images)} real + {len(fake_images)} fake")

# Balance the dataset (undersample fakes to match reals for better training)
random.shuffle(fake_images)
max_per_class = len(real_images)  # Use real count as limit
fake_images_balanced = fake_images[:max_per_class]

all_paths = real_images + fake_images_balanced
all_labels = [0]*len(real_images) + [1]*len(fake_images_balanced)

print(f"Balanced dataset: {len(real_images)} real + {len(fake_images_balanced)} fake = {len(all_paths)} total")

# 80-20 Train-Test split
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels
)

print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")

# ==============================================================================
# DATASET & TRANSFORMS
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

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = SimpleDataset(train_paths, train_labels, train_transforms)
test_dataset = SimpleDataset(test_paths, test_labels, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==============================================================================
# LOAD PRE-TRAINED ROBUST MODEL & FINE-TUNE
# ==============================================================================
# Find model weights
weights_dir = None
for root, dirs, files in os.walk("/kaggle/input/"):
    for f in files:
        if 'Exp2_Robust_best.pth' in f:
            weights_dir = os.path.join(root, f)
            break

if weights_dir is None:
    print("ERROR: Could not find Exp2_Robust_best.pth!")
    print("Searching for any .pth files...")
    import subprocess
    result = subprocess.run(['find', '/kaggle/input/', '-name', '*.pth'], capture_output=True, text=True)
    print(result.stdout)
else:
    print(f"Found weights: {weights_dir}")

# Load model
model = models.efficientnet_b4(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 1)
)
model.load_state_dict(torch.load(weights_dir, map_location=device))
model.to(device)
print("Loaded Exp2 Robust model for fine-tuning!")

# Freeze early layers, only fine-tune the last few layers + classifier
for param in model.features[:6].parameters():
    param.requires_grad = False
print("Frozen early layers. Fine-tuning last layers + classifier.")

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=LEARNING_RATE, weight_decay=1e-4)

best_auc = 0.0
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

print(f"\n{'='*60}")
print("EXPERIMENT 3: Fine-Tuning on Celeb-DF")
print(f"{'='*60}")

for epoch in range(EPOCHS):
    # Train
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
    
    # Evaluate
    model.eval()
    val_loss = 0.0
    all_preds, all_labels_list, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Test]"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels_list.extend(labels.cpu().numpy().flatten())
    
    val_loss /= len(test_loader.dataset)
    val_acc = accuracy_score(all_labels_list, all_preds)
    val_auc = roc_auc_score(all_labels_list, all_probs)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_auc'].append(val_auc)
    
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "/kaggle/working/Exp3_FineTuned_CelebDF_best.pth")
        print("  --> Saved new best model!")

# ==============================================================================
# FINAL EVALUATION & PLOTS
# ==============================================================================
print(f"\n{'='*60}")
print("EXPERIMENT 3 RESULTS")
print(f"{'='*60}")
print(f"Before fine-tuning (Exp2 on Celeb-DF):  Acc=0.3956  AUC=0.7991")
print(f"After fine-tuning (Exp3 on Celeb-DF):   Acc={history['val_acc'][-1]:.4f}  AUC={history['val_auc'][-1]:.4f}")
print(f"{'='*60}")

print(f"\nClassification Report:")
print(classification_report(all_labels_list, all_preds, target_names=['Real', 'Fake']))

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(all_labels_list, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
axes[0].set_title('Exp3: Fine-Tuned on Celeb-DF', fontsize=13)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

# Training curves
epochs_range = range(1, EPOCHS+1)
axes[1].plot(epochs_range, history['val_acc'], 'b-o', label='Accuracy')
axes[1].plot(epochs_range, history['val_auc'], 'g-o', label='AUC')
axes[1].set_title('Fine-Tuning Progress on Celeb-DF', fontsize=13)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('/kaggle/working/exp3_finetuning_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: exp3_finetuning_results.png")

# Summary bar chart comparing all 3 experiments on Celeb-DF
fig, ax = plt.subplots(figsize=(10, 6))
experiments = ['Exp1: Baseline\n(c23 trained)', 'Exp2: Robust\n(mixed trained)', 'Exp3: Fine-Tuned\n(adapted to Celeb-DF)']
accs = [0.4239, 0.3956, history['val_acc'][-1]]
aucs = [0.8285, 0.7991, history['val_auc'][-1]]

x = np.arange(len(experiments))
width = 0.35
bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color=['#e74c3c', '#f39c12', '#2ecc71'])
bars2 = ax.bar(x + width/2, aucs, width, label='AUC-ROC', color=['#c0392b', '#e67e22', '#27ae60'], alpha=0.7)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('All Experiments: Performance on Celeb-DF', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend()
ax.set_ylim(0, 1.15)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/kaggle/working/all_experiments_celebdf.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: all_experiments_celebdf.png")

print(f"\n{'='*60}")
print("ALL EXPERIMENTS COMPLETE!")
print(f"{'='*60}")
print("FF++ Results:")
print(f"  Exp1 Baseline (c23→c40): Acc=0.9280  AUC=0.9938")
print(f"  Exp2 Robust (mixed→c40): Acc=0.9905  AUC=0.9997")
print(f"\nCeleb-DF Results:")
print(f"  Exp1 Baseline (no adaptation): Acc=0.4239  AUC=0.8285")
print(f"  Exp2 Robust (no adaptation):   Acc=0.3956  AUC=0.7991")
print(f"  Exp3 Fine-Tuned (adapted):     Acc={history['val_acc'][-1]:.4f}  AUC={history['val_auc'][-1]:.4f}")
print(f"{'='*60}")
