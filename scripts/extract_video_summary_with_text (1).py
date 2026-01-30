import re
import cv2
import os
import argparse

def extract_segments(text):
    """Extract text descriptions with their frame indices"""
    pattern = r'([^[]+)\[(\d+(?:,\s*\d+)*)\]'
    matches = re.findall(pattern, text)
    segments = []
    for desc, frames in matches:
        indices = [int(x.strip()) for x in frames.split(',')]
        segments.append({'description': desc.strip(), 'frames': indices})
    return segments

def extract_frames_with_text(video_path, segments, output_dir):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_positions = [int(i * total_frames / 100) for i in range(100)]
    
    os.makedirs(output_dir, exist_ok=True)
    saved_frames = []
    
    for segment in segments:
        description = segment['description']
        for idx in segment['frames']:
            if idx >= len(sampled_positions):
                continue
            frame_pos = sampled_positions[idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                # Add text overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, description, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                output_path = os.path.join(output_dir, f'frame_{idx:03d}.jpg')
                cv2.imwrite(output_path, frame)
                saved_frames.append(output_path)
    
    cap.release()
    return saved_frames

def create_summary_video(frame_paths, output_video, fps=2):
    if not frame_paths:
        return
    frame = cv2.imread(frame_paths[0])
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()
    print(f"✅ Summary video with text: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--summary_text", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="video_summary_with_text")
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    
    segments = extract_segments(args.summary_text)
    frames_dir = os.path.join(args.output_dir, "frames")
    frame_paths = extract_frames_with_text(args.video, segments, frames_dir)
    output_video = os.path.join(args.output_dir, "summary_with_text.mp4")
    create_summary_video(frame_paths, output_video, args.fps)
