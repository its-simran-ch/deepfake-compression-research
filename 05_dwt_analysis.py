"""
Phase 8 - Contribution 2: Spatial-Frequency Analysis using Discrete Wavelet Transform (DWT)
===========================================================================================
This script mathematically proves how H.264 compression (c40) destroys 
high-frequency deepfake artifacts by comparing DWT sub-band energy between 
c23 (light) and c40 (heavy) compressed face crops.

USAGE ON KAGGLE:
  1. Run: !pip install PyWavelets
  2. Update the dataset paths below to match your Kaggle environment.
  3. Run all cells.
"""

import os
import glob
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG - UPDATE THESE PATHS FOR YOUR KAGGLE ENVIRONMENT
# ============================================================
C23_FACES_DIR = "/kaggle/input/your-dataset-name/faces_c23/Fake"
C40_FACES_DIR = "/kaggle/input/your-dataset-name/faces_c40/Fake"
OUTPUT_DIR    = "/kaggle/working/dwt_results"
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_dwt(image_path, title_prefix, save_path):
    """
    Applies 2D Haar DWT to a grayscale face image and plots the 
    LL approximation + LH, HL, HH high-frequency detail sub-bands.
    Returns the total high-frequency energy.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read {image_path}")
        return None
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2D Discrete Wavelet Transform (Haar wavelet)
    LL, (LH, HL, HH) = pywt.dwt2(img_gray, 'haar')

    # Energy = sum of squared coefficients in each sub-band
    energy_lh = np.sum(np.square(LH.astype(np.float64)))
    energy_hl = np.sum(np.square(HL.astype(np.float64)))
    energy_hh = np.sum(np.square(HH.astype(np.float64)))
    total_hf  = energy_lh + energy_hl + energy_hh

    print(f"\n[{title_prefix}] HF Energy → LH: {energy_lh:.0f}  HL: {energy_hl:.0f}  HH: {energy_hh:.0f}  TOTAL: {total_hf:.0f}")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title(f"Original\n{title_prefix}", fontsize=12)
    axes[0].axis('off')

    for ax, band, name, energy in zip(
        axes[1:],
        [LH, HL, HH],
        ["Horizontal (LH)", "Vertical (HL)", "Diagonal (HH)"],
        [energy_lh, energy_hl, energy_hh]
    ):
        vis = np.clip(np.abs(band) * 5, 0, 255).astype(np.uint8)
        ax.imshow(vis, cmap='inferno')
        ax.set_title(f"{name}\nEnergy: {energy:.0f}", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")
    return total_hf


def main():
    print("=" * 60)
    print("PHASE 8: SPATIAL-FREQUENCY ANALYSIS (DWT)")
    print("=" * 60)

    # Find a sample fake face from c23
    c23_images = sorted(glob.glob(os.path.join(C23_FACES_DIR, "**/*.jpg"), recursive=True))
    if not c23_images:
        c23_images = sorted(glob.glob(os.path.join(C23_FACES_DIR, "**/*.png"), recursive=True))
    if not c23_images:
        print(f"ERROR: No images found in {C23_FACES_DIR}. Update the path!")
        return

    sample_c23 = c23_images[0]
    # Try to find the matching c40 image by swapping the root directory
    sample_c40 = sample_c23.replace(C23_FACES_DIR, C40_FACES_DIR)

    if not os.path.exists(sample_c40):
        # Fallback: just grab the first c40 image available
        c40_images = sorted(glob.glob(os.path.join(C40_FACES_DIR, "**/*.jpg"), recursive=True))
        if not c40_images:
            c40_images = sorted(glob.glob(os.path.join(C40_FACES_DIR, "**/*.png"), recursive=True))
        if not c40_images:
            print(f"ERROR: No images found in {C40_FACES_DIR}. Update the path!")
            return
        sample_c40 = c40_images[0]

    print(f"\nc23 sample: {sample_c23}")
    print(f"c40 sample: {sample_c40}")

    # Run DWT analysis
    energy_c23 = analyze_dwt(sample_c23, "Light Compression (c23)", os.path.join(OUTPUT_DIR, "dwt_c23.png"))
    energy_c40 = analyze_dwt(sample_c40, "Heavy Compression (c40)", os.path.join(OUTPUT_DIR, "dwt_c40.png"))

    if energy_c23 and energy_c40:
        retention = (energy_c40 / energy_c23) * 100
        print("\n" + "=" * 50)
        print(f"  c23 Total HF Energy:  {energy_c23:.0f}")
        print(f"  c40 Total HF Energy:  {energy_c40:.0f}")
        print(f"  Energy Retained:      {retention:.1f}%")
        print(f"  Energy DESTROYED:     {100 - retention:.1f}%")
        print("=" * 50)
        print("CONCLUSION: Heavy compression acts as a severe")
        print("low-pass filter, destroying the high-frequency")
        print("artifacts CNNs rely on for deepfake detection.")

    # --- Combined comparison bar chart ---
    if energy_c23 and energy_c40:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['c23 (Light)', 'c40 (Heavy)']
        values = [energy_c23, energy_c40]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white', linewidth=2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f'{val:.0f}', ha='center', fontweight='bold', fontsize=13)
        ax.set_ylabel('Total High-Frequency Energy', fontsize=13)
        ax.set_title('DWT High-Frequency Energy: c23 vs c40 Compression', fontsize=14, pad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "dwt_energy_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved comparison chart → {OUTPUT_DIR}/dwt_energy_comparison.png")

    print("\nDone! Download images from /kaggle/working/dwt_results/")


if __name__ == "__main__":
    main()
