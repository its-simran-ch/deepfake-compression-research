"""
Phase 8 - Contribution 1: Explainable AI via Grad-CAM Heatmaps
===============================================================
Copy-paste this entire script into a Kaggle cell.
Make sure you have already run:
  - Cell 1: !pip install grad-cam
  - Cell 2: The model loading code (model = models.efficientnet_b4 ...)
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def run_grad_cam(model, image_path, target_layers, save_path, title, device='cuda'):
    """
    Generates a Grad-CAM heatmap for a given image using the provided model.
    """
    # 1. Load and prepare image
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # BGR → RGB
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img_float = np.float32(rgb_img) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)

    # 2. Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred_label = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else (1 - prob)

    # 3. Generate Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # 4. Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Face", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title("Grad-CAM Activation\n(Red = High Focus)", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(visualization)
    axes[2].set_title(f"Overlay\nPred: {pred_label} ({confidence*100:.1f}%)", fontsize=14)
    axes[2].axis('off')

    plt.suptitle(title, fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved Grad-CAM → {save_path}")


# ============================================================
# RUN GRAD-CAM
# ============================================================
target_layers = [model.features[-1]]

fake_img = '/kaggle/input/datasets/simmi90/ff-extracted-faces/c40/fake/000_003/frame_000.jpg'
real_img = '/kaggle/input/datasets/simmi90/ff-extracted-faces/c23/real/000/frame_000.jpg'

run_grad_cam(model, fake_img, target_layers, '/kaggle/working/gradcam_fake.png',
             'Robust Model Focus: FAKE Face (c40)', device=str(device))
run_grad_cam(model, real_img, target_layers, '/kaggle/working/gradcam_real.png',
             'Robust Model Focus: REAL Face (c23)', device=str(device))
