#!/usr/bin/env python3
"""
Test Stage 3: Frame Coherence Graph
"""

import sys
sys.path.append('/workspace/core-sum-project/src')

import cv2
import numpy as np
import os

def find_video():
    """Find the uploaded video"""
    video_path = "/workspace/core-sum-project/data/Big_Take_2024_-_16mm_Short_Film_1080P.mp4"
    if os.path.exists(video_path):
        return video_path
    return None

def load_frames(video_path, max_frames=600):
    """Load video frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total_frames // max_frames)
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    
    cap.release()
    return frames

def main():
    print("="*60)
    print("TESTING STAGE 3: FRAME COHERENCE GRAPH")
    print("="*60)
    
    # Find video
    video_path = find_video()
    if not video_path:
        print("✗ Video not found")
        return
    
    print(f"\n[1/3] Loading video...")
    frames = load_frames(video_path)
    print(f"    ✓ Loaded {len(frames)} frames")
    
    # Run Stage 1
    print(f"\n[2/3] Running Stage 1 (Hierarchical Pyramid)...")
    from hierarchical_visual_pyramid import HierarchicalVisualPyramid
    hvp = HierarchicalVisualPyramid(device="cuda")
    hierarchy = hvp.encode(frames)
    print(f"    ✓ Scenes: {len(hierarchy.scenes)}, Shots: {len(hierarchy.shots)}")
    
    # Run Stage 2 + 3 (with Graph)
    print(f"\n[3/3] Running Stage 2+3 (CoT + Graph)...")
    from cot_engine import CoTEngine
    cot_engine = CoTEngine(use_graph_refinement=True)
    result = cot_engine.summarize(hierarchy, target_frames=20)
    
    # Show results
    print("\n" + "="*60)
    print("COMPARISON: LLM vs GRAPH-REFINED")
    print("="*60)
    print(f"LLM output: {len(result.coherence_check['final_frame_sequence'])} frames")
    print(f"Graph-refined: {len(result.selected_frames)} frames")
    print(f"\nGraph Metrics:")
    if result.graph_metrics:
        print(f"  - Average coherence: {result.graph_metrics.avg_coherence:.3f}")
        print(f"  - Minimum coherence: {result.graph_metrics.min_coherence:.3f}")
        print(f"  - Redundant frames removed: {len(result.graph_metrics.redundant_frames)}")
        print(f"  - Discontinuous pairs: {len(result.graph_metrics.discontinuous_pairs)}")
        print(f"  - Semantic clusters: {len(result.graph_metrics.semantic_clusters)}")
    print("="*60 + "\n")
    
    print("✓✓✓ STAGE 3 TEST COMPLETE ✓✓✓")

if __name__ == "__main__":
    main()

