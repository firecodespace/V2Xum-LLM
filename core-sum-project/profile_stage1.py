#!/usr/bin/env python3
import sys
sys.path.append('/workspace/core-sum-project/src')

import numpy as np
import time
from hierarchical_visual_pyramid import HierarchicalVisualPyramid

# Generate 600 test frames
print("Generating 600 test frames...")
test_frames = [np.ones((224, 224, 3), dtype=np.uint8) * ((i//150) * 60) for i in range(600)]

print("\n" + "="*60)
print("PROFILING WITH SHARED ENCODER INSTANCES (REALISTIC)")
print("="*60)

# Initialize HVP once (this loads all models)
print("\nInitializing HVP (loading all models)...")
init_start = time.time()
hvp = HierarchicalVisualPyramid(device="cuda")
init_time = time.time() - init_start
print(f"Initialization time: {init_time:.3f}s (one-time cost)\n")

# Now profile the actual encoding (models already loaded)
print("="*60)
print("ENCODING 600 FRAMES (models already loaded)")
print("="*60)

# Scene detection
print("\n[1/3] VideoMAE Scene Detection...")
start = time.time()
scenes = hvp.scene_encoder.detect_scenes(test_frames)
scene_time = time.time() - start
print(f"      Time: {scene_time:.3f}s")
print(f"      Scenes: {len(scenes)}")

# Shot detection
print("\n[2/3] DINOv2 Shot Segmentation...")
start = time.time()
shots = hvp.shot_encoder.detect_shots(test_frames, scenes)
shot_time = time.time() - start
print(f"      Time: {shot_time:.3f}s")
print(f"      Shots: {len(shots)}")

# Frame encoding
print("\n[3/3] CLIP Frame Encoding...")
start = time.time()
embeddings = hvp.frame_encoder.encode_frames(test_frames, batch_size=256)
clip_time = time.time() - start
print(f"      Time: {clip_time:.3f}s")
print(f"      Shape: {embeddings.shape}")

total = scene_time + shot_time + clip_time

print("\n" + "="*60)
print("BREAKDOWN (excluding model loading)")
print("="*60)
print(f"VideoMAE:  {scene_time:.3f}s ({scene_time/total*100:.1f}%)")
print(f"DINOv2:    {shot_time:.3f}s ({shot_time/total*100:.1f}%)")
print(f"CLIP:      {clip_time:.3f}s ({clip_time/total*100:.1f}%)")
print(f"Total:     {total:.3f}s")
print("="*60)

if total <= 1.5:
    print(f"\n✓✓✓ TARGET MET: {total:.3f}s ≤ 1.5s ✓✓✓")
else:
    print(f"\n⚠ Still needs work: {total:.3f}s > 1.5s")
print()
