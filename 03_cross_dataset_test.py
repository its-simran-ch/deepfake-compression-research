"""
Cross-Dataset Generalization Test: Celeb-DF
=============================================
Tests the FF++ trained models on the Celeb-DF dataset.
This is Experiment 3 for the IEEE paper.

INSTRUCTIONS:
1. Create a NEW Kaggle Notebook called "Cross-Dataset Testing"
2. Turn on GPU (T4 x2 or P100), Internet ON
3. Add Input #1: Your "celeb-df" dataset (the one you uploaded)
4. Add Input #2: Your "deepfake-model-training" notebook OUTPUT (for the trained model weights)
   -> Click "+ Add Input" -> "Your Work" -> "Notebooks" -> select "Deepfake Model Training"
5. Cell 1: !pip install facenet-pytorch
6. Cell 2: Paste this entire script
"""

import os
import glob
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from facenet_pytorch import MTCNN
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

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_FRAMES_PER_VIDEO = 15
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# STEP 1: DISCOVER PATHS
# ==============================================================================
def discover_paths():
    """Auto-discover the Celeb-DF and model weights paths"""
    print("Discovering dataset paths...")
    
    # Find Celeb-DF location
    celeb_df_root = None
    model_weights_dir = None
    
    for root, dirs, files in os.walk("/kaggle/input/"):
        depth = root.replace("/kaggle/input/", "").count(os.sep)
        if depth > 4:
            continue
        
        # Look for Celeb-DF markers
        for d in dirs:
            if d.lower() in ['celeb-real', 'celeb-synthesis', 'celeb_real', 'celeb_synthesis',
                             'celeb-df-v2', 'celebdf', 'celeb_df']:
                celeb_df_root = root
                break
        
        # Look for model weights
        for f in files:
            if f.endswith('_best.pth'):
                model_weights_dir = root
    
    return celeb_df_root, model_weights_dir

# ==============================================================================
# STEP 2: EXPLORE CELEB-DF STRUCTURE
# ==============================================================================
def explore_celebdf(root_path):
    """Show the Celeb-DF folder structure"""
    print(f"\nCeleb-DF root: {root_path}")
    print("Structure:")
    for item in sorted(os.listdir(root_path)):
        full_path = os.path.join(root_path, item)
        if os.path.isdir(full_path):
            count = len(os.listdir(full_path))
            print(f"  📁 {item}/ ({count} items)")
        else:
            print(f"  📄 {item}")
    
    # Try to find real and fake video directories
    real_dirs = []
    fake_dirs = []
    
    for item in os.listdir(root_path):
        item_lower = item.lower().replace('-', '_').replace(' ', '_')
        full_path = os.path.join(root_path, item)
        if not os.path.isdir(full_path):
            continue
        if 'real' in item_lower or 'youtube' in item_lower:
            real_dirs.append(full_path)
        elif 'synthesis' in item_lower or 'fake' in item_lower or 'swap' in item_lower:
            fake_dirs.append(full_path)
    
    return real_dirs, fake_dirs

# ==============================================================================
# STEP 3: EXTRACT FACES FROM CELEB-DF
# ==============================================================================
def extract_faces_from_videos(video_dirs, output_dir, label_name):
    """Extract faces from a list of video directories"""
    mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)
    
    all_video_paths = []
    for vdir in video_dirs:
        all_video_paths.extend(glob.glob(os.path.join(vdir, "*.mp4")))
        all_video_paths.extend(glob.glob(os.path.join(vdir, "*.avi")))
    
    print(f"\n  Extracting faces from {len(all_video_paths)} {label_name} videos...")
    out_dir = os.path.join(output_dir, label_name)
    os.makedirs(out_dir, exist_ok=True)
    
    for video_path in tqdm(all_video_paths, desc=f"{label_name}"):
        video_name = os.path.basename(video_path).split('.')[0]
        vid_out_dir = os.path.join(out_dir, video_name)
        
        # Resume support
        if os.path.exists(vid_out_dir) and len(os.listdir(vid_out_dir)) >= NUM_FRAMES_PER_VIDEO:
            continue
        
        os.makedirs(vid_out_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            continue
        
        interval = max(1, frame_count // NUM_FRAMES_PER_VIDEO)
        count = 0
        extracted = 0
        
        while cap.isOpened() and extracted < NUM_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                save_path = os.path.join(vid_out_dir, f"frame_{extracted:03d}.jpg")
                try:
                    face = mtcnn(img, save_path=save_path)
                    if face is not None:
                        extracted += 1
                except Exception:
                    pass
            count += 1
        cap.release()
    
    # Count extracted images
    all_images = glob.glob(os.path.join(out_dir, "*", "*.jpg"))
    print(f"  {label_name}: {len(all_images)} face images extracted")
    return all_images

# ==============================================================================
# STEP 4: LOAD TRAINED MODEL
# ==============================================================================
def load_model(weights_path):
    """Load a pre-trained EfficientNet-B4 with saved weights"""
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Loaded model from {os.path.basename(weights_path)}")
    return model

# ==============================================================================
# STEP 5: EVALUATE ON CELEB-DF
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

def evaluate_model(model, dataloader, name):
    """Run inference and compute metrics"""
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
def main():
    # Step 1: Discover paths
    celeb_root, weights_dir = discover_paths()
    
    if celeb_root is None:
        print("ERROR: Could not find Celeb-DF dataset!")
        print("Listing /kaggle/input/ contents:")
        for root, dirs, files in os.walk("/kaggle/input/"):
            level = root.replace("/kaggle/input/", "").count(os.sep)
            if level <= 3:
                print(f"{'  '*level}{os.path.basename(root)}/  ({len(files)} files)")
        return
    
    if weights_dir is None:
        print("ERROR: Could not find trained model weights!")
        print("Make sure to add the 'Deepfake Model Training' notebook output as Input.")
        return
    
    # Step 2: Explore structure
    real_dirs, fake_dirs = explore_celebdf(celeb_root)
    print(f"\n  Real video dirs: {[os.path.basename(d) for d in real_dirs]}")
    print(f"  Fake video dirs: {[os.path.basename(d) for d in fake_dirs]}")
    
    if not real_dirs or not fake_dirs:
        print("ERROR: Could not identify real/fake directories in Celeb-DF.")
        print("Please check the folder names above and report them.")
        return
    
    # Step 3: Extract faces
    output_dir = "/kaggle/working/celebdf_faces"
    real_images = extract_faces_from_videos(real_dirs, output_dir, "real")
    fake_images = extract_faces_from_videos(fake_dirs, output_dir, "fake")
    
    # Prepare dataset
    all_paths = real_images + fake_images
    all_labels = [0]*len(real_images) + [1]*len(fake_images)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = SimpleDataset(all_paths, all_labels, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\nCeleb-DF test set: {len(real_images)} real + {len(fake_images)} fake = {len(all_paths)} total")
    
    # Step 4: Load and evaluate both models
    print("\n" + "="*60)
    print("CROSS-DATASET EVALUATION ON CELEB-DF")
    print("="*60)
    
    # Find model weight files
    exp1_weights = os.path.join(weights_dir, "Exp1_Baseline_best.pth")
    exp2_weights = os.path.join(weights_dir, "Exp2_Robust_best.pth")
    
    if not os.path.exists(exp1_weights):
        # Try alternate names
        for f in os.listdir(weights_dir):
            if 'baseline' in f.lower() or 'exp1' in f.lower():
                exp1_weights = os.path.join(weights_dir, f)
            if 'robust' in f.lower() or 'exp2' in f.lower():
                exp2_weights = os.path.join(weights_dir, f)
    
    model1 = load_model(exp1_weights)
    acc1, auc1, labels1, preds1 = evaluate_model(model1, test_loader, "Exp1 Baseline (c23 trained)")
    
    model2 = load_model(exp2_weights)
    acc2, auc2, labels2, preds2 = evaluate_model(model2, test_loader, "Exp2 Robust (mixed trained)")
    
    # Step 5: Plot results
    print("\n" + "="*60)
    print("CROSS-DATASET RESULTS SUMMARY")
    print("="*60)
    print(f"Exp1 Baseline on Celeb-DF:  Acc={acc1:.4f}  AUC={auc1:.4f}")
    print(f"Exp2 Robust on Celeb-DF:    Acc={acc2:.4f}  AUC={auc2:.4f}")
    print("="*60)
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm1 = confusion_matrix(labels1, preds1)
    cm2 = confusion_matrix(labels2, preds2)
    
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    axes[0].set_title('Exp1: Baseline on Celeb-DF', fontsize=13)
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    axes[1].set_title('Exp2: Robust on Celeb-DF', fontsize=13)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/celebdf_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: celebdf_confusion_matrices.png")
    
    # Summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, [acc1, acc2], width, label='Accuracy', color=['#e74c3c', '#2ecc71'])
    bars2 = ax.bar(x + width/2, [auc1, auc2], width, label='AUC-ROC', color=['#c0392b', '#27ae60'], alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_title('Cross-Dataset Generalization: Celeb-DF', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Exp1: Baseline', 'Exp2: Robust'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('/kaggle/working/celebdf_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: celebdf_comparison.png")
    
    print("\nAll Celeb-DF results saved to /kaggle/working/")

if __name__ == "__main__":
    main()
