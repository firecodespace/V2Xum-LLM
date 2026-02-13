#!/usr/bin/env python3
"""
Test V2Xum-LLM on a real video
Downloads and processes a sample video using Python only
"""

import sys
import os
sys.path.append('/workspace/core-sum-project/src')

import cv2
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm

def download_sample_video():
    """Download a short sample video (Python only)"""
    
    print("\n[1/5] Downloading sample video...")
    
    # Use a smaller, direct-download sample video
    # Sample from sample-videos.com (10MB, 30s)
    url = "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"
    
    video_path = "/workspace/core-sum-project/data/sample_video.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    print(f"    Downloading from: sample-videos.com")
    
    try:
        # Download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(video_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc='    ') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        file_size = os.path.getsize(video_path)
        print(f"    ✓ Downloaded {file_size/1024/1024:.1f}MB")
        print(f"    ✓ Saved to: {video_path}")
        
        return video_path
        
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        print(f"    Trying alternative source...")
        
        # Fallback: smaller sample
        url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(video_path)
            print(f"    ✓ Downloaded {file_size/1024/1024:.1f}MB (fallback)")
            return video_path
            
        except Exception as e2:
            print(f"    ✗ Fallback also failed: {e2}")
            return None

def load_video_frames(video_path, max_frames=600, sample_rate=1):
    """Load frames from video file"""
    
    print(f"\n[2/5] Loading video frames...")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"    ✗ Failed to open video")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"    Video info:")
    print(f"    - Resolution: {width}x{height}")
    print(f"    - FPS: {fps:.1f}")
    print(f"    - Total frames: {total_frames}")
    print(f"    - Duration: {duration:.1f}s")
    
    frames = []
    frame_idx = 0
    loaded_count = 0
    
    # Sample frames to limit to max_frames
    if total_frames > max_frames:
        # Sample evenly across video
        sample_rate = total_frames / max_frames
        print(f"    - Sampling every {sample_rate:.1f} frames")
    
    with tqdm(total=min(max_frames, total_frames), desc="    Loading") as pbar:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if sample_rate > 1:
                if frame_idx % int(sample_rate) == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    loaded_count += 1
                    pbar.update(1)
            else:
                # Load all frames
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                loaded_count += 1
                pbar.update(1)
            
            frame_idx += 1
    
    cap.release()
    
    actual_fps = len(frames) / duration if duration > 0 else fps
    
    print(f"    ✓ Loaded {len(frames)} frames (~{actual_fps:.1f} effective FPS)")
    return frames

def run_stage1(frames):
    """Run Stage 1: Hierarchical Visual Pyramid"""
    
    print(f"\n[3/5] Running Stage 1: Hierarchical Visual Pyramid...")
    
    from hierarchical_visual_pyramid import HierarchicalVisualPyramid
    
    hvp = HierarchicalVisualPyramid(device="cuda")
    hierarchy = hvp.encode(frames)
    
    print(f"    ✓ Scenes: {len(hierarchy.scenes)}")
    print(f"    ✓ Shots: {len(hierarchy.shots)}")
    print(f"    ✓ Frame embeddings: {hierarchy.frames.shape}")
    
    return hierarchy

def run_stage2(hierarchy, target_frames=20):
    """Run Stage 2: Chain-of-Thought Reasoning"""
    
    print(f"\n[4/5] Running Stage 2: Chain-of-Thought Reasoning...")
    print(f"    Target: {target_frames} frames")
    
    from cot_engine import CoTEngine
    
    cot_engine = CoTEngine()
    result = cot_engine.summarize(hierarchy, target_frames=target_frames)
    
    print(f"    ✓ Selected {len(result.selected_frames)} frames")
    print(f"    ✓ Coherence: {result.coherence_check['is_coherent']}")
    
    return result

def visualize_results(frames, hierarchy, cot_result):
    """Create visual summary"""
    
    print(f"\n[5/5] Generating visual summary...")
    
    output_dir = "/workspace/core-sum-project/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save selected keyframes
    saved_count = 0
    for i, frame_idx in enumerate(cot_result.selected_frames):
        if frame_idx < len(frames):
            frame = frames[frame_idx]
            output_path = f"{output_dir}/keyframe_{i:03d}_frame{frame_idx:04d}.jpg"
            
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, frame_bgr)
            saved_count += 1
    
    print(f"    ✓ Saved {saved_count} keyframes to {output_dir}/")
    
    # Print scene breakdown
    print(f"\n{'='*60}")
    print(f"SCENE ANALYSIS")
    print(f"{'='*60}")
    
    for ranking in cot_result.scene_rankings:
        scene_id = ranking['scene_id']
        importance = ranking['importance']
        reason = ranking['reason']
        
        # Count frames in this scene
        scene_start, scene_end = hierarchy.scenes[scene_id]
        frames_in_scene = [
            f for f in cot_result.selected_frames
            if scene_start <= f < scene_end
        ]
        
        print(f"\nScene {scene_id} (frames {scene_start}-{scene_end}):")
        print(f"  Importance: {importance}/10")
        print(f"  Reason: {reason}")
        print(f"  Selected frames: {len(frames_in_scene)}")
        if frames_in_scene:
            print(f"  Indices: {frames_in_scene[:5]}" + 
                  (f"... (+{len(frames_in_scene)-5} more)" if len(frames_in_scene) > 5 else ""))
    
    print(f"\n{'='*60}\n")

def main():
    print("="*60)
    print("TESTING V2Xum-LLM ON REAL VIDEO")
    print("="*60)
    
    # Download video
    video_path = download_sample_video()
    if not video_path or not os.path.exists(video_path):
        print("✗ Failed to download video")
        return
    
    # Load frames
    frames = load_video_frames(video_path, max_frames=600)
    if not frames:
        print("✗ Failed to load frames")
        return
    
    # Run Stage 1
    hierarchy = run_stage1(frames)
    
    # Run Stage 2
    cot_result = run_stage2(hierarchy, target_frames=20)
    
    # Visualize
    visualize_results(frames, hierarchy, cot_result)
    
    print("="*60)
    print("✓✓✓ TEST COMPLETE ✓✓✓")
    print("="*60)
    print(f"\nKeyframes saved to: /workspace/core-sum-project/output/")
    print(f"Total frames selected: {len(cot_result.selected_frames)}")
    print(f"\nYou can view the keyframes to see the summary quality!")

if __name__ == "__main__":
    main()

