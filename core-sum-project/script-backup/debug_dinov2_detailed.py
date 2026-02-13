import sys
sys.path.append('/workspace/core-sum-project/src')
import numpy as np
import time
import torch
import cv2

print("\n" + "="*60)
print("TESTING NEW DISTANCE-BASED SHOT DETECTION")
print("="*60)

frames = [np.ones((224, 224, 3), dtype=np.uint8) for _ in range(600)]
scenes = [(0, 150), (150, 300), (300, 450), (450, 600)]

print("\n[1] Loading model...")
start = time.time()
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True, verbose=False)
model = model.to('cuda').half().eval()
load_time = time.time() - start

print(f"    Time: {load_time:.3f}s")

# Collect samples
all_samples = []
for scene_start, scene_end in scenes:
    indices = np.linspace(scene_start, scene_end-1, 5, dtype=int)
    for idx in indices:
        all_samples.append(frames[idx])

print(f"\n[2] Processing {len(all_samples)} samples...")
start = time.time()

# Resize + convert
resized = np.stack([cv2.resize(f, (224, 224)) for f in all_samples])
batch_tensor = torch.from_numpy(resized).permute(0, 3, 1, 2).float()
batch_tensor = batch_tensor.div_(255.0).half().to('cuda', non_blocking=True)

# Inference
with torch.no_grad():
    features = model(batch_tensor).cpu().numpy()

# Distance-based split (NO sklearn)
for i in range(4):
    scene_features = features[i*5:(i+1)*5]
    
    # Find max distance point
    max_dist = 0
    split_idx = 2
    for j in range(1, 5):
        dist = np.linalg.norm(scene_features[j] - scene_features[j-1])
        if dist > max_dist:
            max_dist = dist
            split_idx = j

process_time = time.time() - start
print(f"    Time: {process_time:.3f}s")

print("\n" + "="*60)
print(f"Total (excluding model load): {process_time:.3f}s")
print("="*60 + "\n")
