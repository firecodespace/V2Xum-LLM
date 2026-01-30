import re
import cv2
import os
import argparse
from pathlib import Path

def extract_frame_indices(text):
    """Extract frame indices from model output like [10, 11, 12, 13]"""
    pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    matches = re.findall(pattern, text)
    frame_indices = []
    for match in matches:
        indices = [int(x.strip()) for x in match.split(',')]
        frame_indices.extend(indices)
    return sorted(set(frame_indices))

def extract_frames(video_path, frame_indices, output_dir):
    """Extract specific frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample 100 frames uniformly (as model does)
    sampled_positions = [int(i * total_frames / 100) for i in range(100)]
    
    os.makedirs(output_dir, exist_ok=True)
    saved_frames = []
    
    for idx in frame_indices:
        if idx >= len(sampled_positions):
            continue
        frame_pos = sampled_positions[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f'frame_{idx:03d}.jpg')
            cv2.imwrite(output_path, frame)
            saved_frames.append(output_path)
            print(f"Saved frame {idx} -> {output_path}")
    
    cap.release()
    return saved_frames

def create_summary_video(frame_paths, output_video, fps=2):
    """Create video from extracted frames"""
    if not frame_paths:
        print("No frames to create video")
        return
    
    frame = cv2.imread(frame_paths[0])
    height, width = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"✅ Summary video created: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--summary_text", type=str, required=True, help="Model output text with frame indices")
    parser.add_argument("--output_dir", type=str, default="video_summary_output")
    parser.add_argument("--fps", type=int, default=2, help="FPS for summary video")
    args = parser.parse_args()
    
    # Extract frame indices
    frame_indices = extract_frame_indices(args.summary_text)
    print(f"Extracted frame indices: {frame_indices}")
    
    # Create output directory
    frames_dir = os.path.join(args.output_dir, "frames")
    
    # Extract frames
    frame_paths = extract_frames(args.video, frame_indices, frames_dir)
    
    # Create summary video
    output_video = os.path.join(args.output_dir, "summary_video.mp4")
    create_summary_video(frame_paths, output_video, args.fps)
    
    print(f"\n✅ Video summary complete!")
    print(f"   Frames saved to: {frames_dir}")
    print(f"   Summary video: {output_video}")
