#!/usr/bin/env python3
"""Test Stage 2 Chain-of-Thought Reasoning"""

import sys
sys.path.append('/workspace/core-sum-project/src')

import numpy as np
from hierarchical_visual_pyramid import HierarchicalVisualPyramid
from cot_engine import CoTEngine

print("\n" + "="*60)
print("TESTING STAGE 2: CHAIN-OF-THOUGHT REASONING")
print("="*60)

# Generate test video (600 frames, 4 scenes)
print("\n[1] Generating test video...")
test_frames = []
for i in range(600):
    scene_id = i // 150
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
    frame = np.ones((224, 224, 3), dtype=np.uint8) * np.array(colors[scene_id], dtype=np.uint8)
    test_frames.append(frame)
print("    ✓ 600 frames created")

# Run Stage 1
print("\n[2] Running Stage 1: Hierarchical Visual Pyramid...")
hvp = HierarchicalVisualPyramid(device="cuda")
hierarchy = hvp.encode(test_frames)
print(f"    ✓ Scenes: {len(hierarchy.scenes)}")
print(f"    ✓ Shots: {len(hierarchy.shots)}")
print(f"    ✓ Frames: {hierarchy.frames.shape}")

# Run Stage 2
print("\n[3] Running Stage 2: Chain-of-Thought Reasoning...")
cot_engine = CoTEngine()
result = cot_engine.summarize(hierarchy, target_frames=20)

# Display results
print("\n" + "="*60)
print("STAGE 2 RESULTS")
print("="*60)
print(f"Selected frames: {len(result.selected_frames)}")
print(f"Frame indices: {result.selected_frames[:10]}..." if len(result.selected_frames) > 10 else f"Frame indices: {result.selected_frames}")
print(f"\nScene rankings:")
for ranking in result.scene_rankings:
    print(f"  Scene {ranking['scene_id']}: {ranking['importance']}/10 - {ranking['reason']}")

print(f"\nCoherence: {'✓ PASS' if result.coherence_check['is_coherent'] else '✗ FAIL'}")

print("\n" + "="*60)
print("✓✓✓ STAGE 2 TEST COMPLETE ✓✓✓")
print("="*60 + "\n")
