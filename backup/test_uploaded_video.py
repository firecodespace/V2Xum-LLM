#!/usr/bin/env python3
"""
Test V2Xum-LLM on user-uploaded video
"""

import sys
import os
sys.path.append('/workspace/core-sum-project/src')

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def find_video_file():
    """Look for video file in common locations"""
    
    print("\n[1/5] Looking for video file...")
    
    # Check if specific video exists
    specific_path = "/workspace/core-sum-project/data/Big_Take_2024_-_16mm_Short_Film_1080P.mp4"
    if os.path.exists(specific_path):
        print(f"    ✓ Found video: {specific_path}")
        return specific_path
    
    # Common locations to check
    search_paths = [
        "/workspace/core-sum-project/data",
        "/workspace",
        "/workspace/core-sum-project",
        ".",
    ]
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(root, file)
                    print(f"    ✓ Found video: {video_path}")
                    return video_path
    
    print(f"    ✗ No video file found")
    return None

def load_video_frames(video_path, max_frames=600):
    """Load frames from video file with smart sampling"""
    
    print(f"\n[2/5] Loading video frames...")
    print(f"    File: {os.path.basename(video_path)}")
    
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
    print(f"    - Duration: {duration:.1f}s ({duration/60:.1f}min)")
    
    # Calculate sampling rate
    if total_frames <= max_frames:
        sample_rate = 1
        print(f"    - Loading all {total_frames} frames")
    else:
        sample_rate = total_frames / max_frames
        print(f"    - Sampling every {sample_rate:.1f} frames (to get ~{max_frames} frames)")
    
    frames = []
    frame_idx = 0
    
    with tqdm(total=min(max_frames, total_frames), desc="    Loading", unit="frame") as pbar:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % max(1, int(sample_rate)) == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                pbar.update(1)
            
            frame_idx += 1
    
    cap.release()
    
    effective_fps = len(frames) / duration if duration > 0 else fps
    
    print(f"    ✓ Loaded {len(frames)} frames")
    print(f"    ✓ Effective FPS: {effective_fps:.1f}")
    print(f"    ✓ Coverage: {len(frames)/total_frames*100:.1f}% of video")
    
    return frames

def run_stage1(frames):
    """Run Stage 1: Hierarchical Visual Pyramid"""
    
    print(f"\n[3/5] Running Stage 1: Hierarchical Visual Pyramid...")
    
    from hierarchical_visual_pyramid import HierarchicalVisualPyramid
    
    hvp = HierarchicalVisualPyramid(device="cuda")
    hierarchy = hvp.encode(frames)
    
    print(f"    ✓ Scenes detected: {len(hierarchy.scenes)}")
    print(f"    ✓ Shots detected: {len(hierarchy.shots)}")
    print(f"    ✓ Frame embeddings: {hierarchy.frames.shape}")
    
    # Show scene breakdown
    print(f"\n    Scene breakdown:")
    for i, (start, end) in enumerate(hierarchy.scenes):
        duration = end - start
        print(f"      Scene {i}: frames {start:4d}-{end:4d} ({duration:3d} frames)")
    
    return hierarchy

def run_stage2(hierarchy, target_frames=20):
    """Run Stage 2: Chain-of-Thought Reasoning"""
    
    print(f"\n[4/5] Running Stage 2: Chain-of-Thought Reasoning...")
    print(f"    Target: {target_frames} frames")
    
    from cot_engine import CoTEngine
    
    cot_engine = CoTEngine()
    result = cot_engine.summarize(hierarchy, target_frames=target_frames)
    
    print(f"    ✓ Selected {len(result.selected_frames)} frames")
    print(f"    ✓ Target accuracy: {len(result.selected_frames)/target_frames*100:.1f}%")
    print(f"    ✓ Coherence: {'PASS' if result.coherence_check['is_coherent'] else 'FIXED (added bridges)'}")
    
    return result

def visualize_results(frames, hierarchy, cot_result):
    """Create visual summary and analysis"""
    
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
            
            # Add frame number overlay
            cv2.putText(
                frame_bgr, 
                f"Frame {frame_idx}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            cv2.imwrite(output_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
    
    print(f"    ✓ Saved {saved_count} keyframes to {output_dir}/")
    
    # Print detailed scene analysis
    print(f"\n{'='*60}")
    print(f"DETAILED SCENE ANALYSIS")
    print(f"{'='*60}")
    
    for ranking in cot_result.scene_rankings:
        scene_id = ranking['scene_id']
        importance = ranking['importance']
        reason = ranking['reason']
        
        # Get scene bounds
        scene_start, scene_end = hierarchy.scenes[scene_id]
        scene_duration = scene_end - scene_start
        
        # Count frames in this scene
        frames_in_scene = [
            f for f in cot_result.selected_frames
            if scene_start <= f < scene_end
        ]
        
        # Calculate coverage
        coverage = len(frames_in_scene) / scene_duration * 100 if scene_duration > 0 else 0
        
        print(f"\n{'─'*60}")
        print(f"Scene {scene_id}:")
        print(f"  Range: frames {scene_start}-{scene_end} ({scene_duration} frames)")
        print(f"  Importance: {importance}/10 {'★'*importance}")
        print(f"  Reason: {reason}")
        print(f"  Selected: {len(frames_in_scene)} keyframes ({coverage:.1f}% coverage)")
        
        if frames_in_scene:
            # Show frame distribution
            frame_list = ', '.join([str(f) for f in frames_in_scene[:10]])
            if len(frames_in_scene) > 10:
                frame_list += f" ... (+{len(frames_in_scene)-10} more)"
            print(f"  Frames: [{frame_list}]")
    
    # Show coherence details
    print(f"\n{'='*60}")
    print(f"COHERENCE CHECK")
    print(f"{'='*60}")
    print(f"Status: {'✓ COHERENT' if cot_result.coherence_check['is_coherent'] else '⚠ GAPS DETECTED'}")
    
    if not cot_result.coherence_check['is_coherent']:
        issues = cot_result.coherence_check.get('issues', [])
        additions = cot_result.coherence_check.get('suggested_additions', [])
        
        if issues:
            print(f"\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        if additions:
            print(f"\nBridging frames added: {len(additions)}")
            for add in additions:
                print(f"  - Frame {add['frame_index']}: {add['reason']}")
    
    print(f"\n{'='*60}\n")

def main():
    print("="*60)
    print("V2Xum-LLM: REAL VIDEO TEST")
    print("="*60)
    
    # Find video file
    video_path = find_video_file()
    
    if not video_path:
        print("\n✗ No video found")
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
    print(f"\nResults:")
    print(f"  - Input video: {os.path.basename(video_path)}")
    print(f"  - Frames processed: {len(frames)}")
    print(f"  - Scenes detected: {len(hierarchy.scenes)}")
    print(f"  - Keyframes selected: {len(cot_result.selected_frames)}")
    print(f"  - Output location: /workspace/core-sum-project/output/")
    print(f"\n💡 View the saved keyframes to see the summary quality!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

