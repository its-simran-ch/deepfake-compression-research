"""
Face Extraction Script for FaceForensics++ Dataset on Kaggle
=============================================================
This script extracts faces from FF++ videos using MTCNN face detection.
It processes both c23 (light compression) and c40 (heavy compression) videos.

INSTRUCTIONS:
1. Create a NEW Kaggle Notebook
2. Turn on GPU (T4 x2 or P100) in Settings
3. Click "+ Add Input" -> "Your Work" -> "Datasets" -> select "faceforensics-research-data"
4. Paste this entire script into a code cell and run it
"""

import os
import cv2
import glob
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION — These paths match YOUR Kaggle dataset structure exactly
# ==============================================================================
NUM_FRAMES_PER_VIDEO = 15  # Evenly spaced frames per video

# This is where Kaggle mounts your private dataset
DATASET_ROOT = "/kaggle/input/datasets/simranch77/faceforensics-research-data/faceforensics"

# Output directory (Kaggle's writable area)
OUTPUT_ROOT = "/kaggle/working/extracted_faces"

# Define the exact folder paths based on the FF++ structure you downloaded
VIDEO_PATHS = {
    "c23_real": f"{DATASET_ROOT}/original_sequences/youtube/c23/videos",
    "c40_real": f"{DATASET_ROOT}/original_sequences/youtube/c40/videos",
    "c23_fake": f"{DATASET_ROOT}/manipulated_sequences/Deepfakes/c23/videos",
    "c40_fake": f"{DATASET_ROOT}/manipulated_sequences/Deepfakes/c40/videos",
}

# ==============================================================================
# INITIALIZATION
# ==============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)

# ==============================================================================
# VIDEO PROCESSING FUNCTION
# ==============================================================================
def process_video(video_path, output_dir):
    """Extracts evenly spaced frames from a video and crops the face."""
    video_name = os.path.basename(video_path).split('.')[0]
    out_dir = os.path.join(output_dir, video_name)
    
    # Resume support: skip if already processed
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) >= NUM_FRAMES_PER_VIDEO:
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return
    
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
            save_path = os.path.join(out_dir, f"frame_{extracted:03d}.jpg")
            try:
                face = mtcnn(img, save_path=save_path)
                if face is not None:
                    extracted += 1
            except Exception:
                pass
                
        count += 1
        
    cap.release()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    for key, video_dir in VIDEO_PATHS.items():
        compression, label = key.split("_")  # e.g., "c23", "real"
        
        print(f"\n--- Processing: {label.upper()} videos ({compression}) ---")
        
        # Check if the path exists
        if not os.path.exists(video_dir):
            # Try alternate path (singular vs plural)
            alt_dir = video_dir.replace("manipulated_sequences", "manipulated_sequence")
            if os.path.exists(alt_dir):
                video_dir = alt_dir
            else:
                print(f"  WARNING: Path not found: {video_dir}")
                print(f"  Please check the folder structure in your Kaggle Input panel.")
                continue
        
        video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
        if len(video_paths) == 0:
            print(f"  WARNING: No .mp4 files found in {video_dir}")
            continue
            
        print(f"  Found {len(video_paths)} videos")
        
        output_dir = os.path.join(OUTPUT_ROOT, compression, label)
        os.makedirs(output_dir, exist_ok=True)
        
        for video_path in tqdm(video_paths, desc=f"{label} ({compression})"):
            process_video(video_path, output_dir)
                
    print(f"\n{'='*60}")
    print(f"Extraction Complete! Data saved to {OUTPUT_ROOT}")
    print(f"{'='*60}")
    
    # Print summary
    for compression in ['c23', 'c40']:
        for label in ['real', 'fake']:
            path = os.path.join(OUTPUT_ROOT, compression, label)
            if os.path.exists(path):
                count = len(os.listdir(path))
                print(f"  {compression}/{label}: {count} video folders")

if __name__ == "__main__":
    main()
