"""
Baseline diagnostic script to evaluate my_test.mp4.
Analyzes detection confidences, bounding box sizes/aspect ratios, tracking IDs,
and ID continuity across frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import defaultdict

def analyze_video(video_path="videos/my_test.mp4", model_name="yolov8n.pt", conf=0.35, tracker_cfg="bytetrack.yaml"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n==========================================")
    print(f"ANALYSIS: {video_path}")
    print(f"Model: {model_name} | Conf: {conf} | Tracker: {tracker_cfg}")
    print(f"Video: {width}x{height} @ {fps:.1f} fps | {total_frames} frames")
    print(f"==========================================")
    
    model = YOLO(model_name)
    
    track_history = defaultdict(list)  # track_id -> [(frame, bbox, conf, center)]
    frame_detections = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            conf=conf,
            classes=[0],
            verbose=False
        )
        
        frame_data = {"frame": frame_idx, "workers": []}
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                score = float(boxes.conf[i].cpu().numpy())
                track_id = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else -1
                
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                bw, bh = x2 - x1, y2 - y1
                aspect_ratio = bh / max(bw, 1e-5)
                
                item = {
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "conf": score,
                    "aspect_ratio": aspect_ratio
                }
                frame_data["workers"].append(item)
                if track_id >= 0:
                    track_history[track_id].append((frame_idx, item))
                    
        frame_detections.append(frame_data)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
            
    cap.release()
    
    print("\n--- TRACK SUMMARY ---")
    print(f"Total Unique Track IDs: {len(track_history)}")
    
    # Sort tracks by duration
    sorted_tracks = sorted(track_history.items(), key=lambda x: len(x[1]), reverse=True)
    
    for tid, frames in sorted_tracks:
        start_f = frames[0][0]
        end_f = frames[-1][0]
        duration_frames = len(frames)
        span_frames = end_f - start_f + 1
        avg_conf = np.mean([f[1]["conf"] for f in frames])
        avg_aspect = np.mean([f[1]["aspect_ratio"] for f in frames])
        # Calculate stationary vs moving
        centers = np.array([f[1]["center"] for f in frames])
        displacement = np.linalg.norm(centers[-1] - centers[0]) if len(centers) > 1 else 0
        
        print(f"Track ID {tid:3d}: {duration_frames:4d} frames (span: {start_f:4d}-{end_f:4d}) | "
              f"avg conf: {avg_conf:.3f} | avg aspect: {avg_aspect:.2f} | "
              f"start bbox: {[round(c,1) for c in frames[0][1]['bbox']]} | "
              f"displacement: {displacement:.1f}px")
              
    return track_history, frame_detections

if __name__ == "__main__":
    analyze_video()
