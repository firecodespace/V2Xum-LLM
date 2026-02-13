#!/usr/bin/env python3
"""Download all required models for Stage 1"""

import torch
from transformers import CLIPModel, CLIPProcessor, VideoMAEForVideoClassification, VideoMAEImageProcessor
import os

os.makedirs("models", exist_ok=True)

print("\n" + "="*60)
print("DOWNLOADING MODELS FOR STAGE 1")
print("="*60)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

# Download CLIP
print("\n[1/3] Downloading CLIP ViT-L/14...")
print("      Size: ~1.7 GB (this will take 3-5 minutes)")
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.save_pretrained("./models/clip-vit-large-patch14")
    clip_processor.save_pretrained("./models/clip-vit-large-patch14")
    print("      ✓ CLIP downloaded successfully")
except Exception as e:
    print(f"      ✗ Error: {e}")
    exit(1)

# Download VideoMAE
print("\n[2/3] Downloading VideoMAE-Base...")
print("      Size: ~360 MB (this will take 1-2 minutes)")
try:
    videomae_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
    videomae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    videomae_model.save_pretrained("./models/videomae-base")
    videomae_processor.save_pretrained("./models/videomae-base")
    print("      ✓ VideoMAE downloaded successfully")
except Exception as e:
    print(f"      ✗ Error: {e}")
    exit(1)

# DINOv2 info (loaded via torch.hub)
print("\n[3/3] DINOv2 ViT-B/14...")
print("      Size: ~330 MB")
print("      Note: Will auto-download on first use via torch.hub")
print("      ✓ Configured")

print("\n" + "="*60)
print("✓ ALL MODELS READY")
print("="*60)
print("\nModel locations:")
print("  - CLIP:     ./models/clip-vit-large-patch14/")
print("  - VideoMAE: ./models/videomae-base/")
print("  - DINOv2:   (will download to ~/.cache/torch/hub/)")
print("\nTotal disk space: ~2.4 GB")
print("\n" + "="*60)
