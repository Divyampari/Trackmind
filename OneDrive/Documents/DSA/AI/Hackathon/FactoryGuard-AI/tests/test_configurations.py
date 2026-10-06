"""
Comprehensive configuration evaluator for FactoryGuard AI Phase 1.
Tests various combinations of:
- Models: yolov8n.pt vs yolov8s.pt
- Confidence thresholds: 0.25, 0.35, 0.45, 0.50
- ByteTrack configurations: default vs tuned
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import yaml

def create_tracker_configs():
    os.makedirs("models/trackers", exist_ok=True)
    
    # Tuned ByteTrack config A: Moderate buffer (60 frames), higher new track threshold
    cfg_a = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.40,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.50,
        "track_buffer": 60,
        "match_thresh": 0.80,
        "fuse_score": True,
    }
    with open("models/trackers/bytetrack_tuned_60.yaml", "w") as f:
        yaml.dump(cfg_a, f)
        
    # Tuned ByteTrack config B: Extended buffer (90 frames), higher match threshold
    cfg_b = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.45,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.55,
        "track_buffer": 90,
        "match_thresh": 0.85,
        "fuse_score": True,
    }
    with open("models/trackers/bytetrack_tuned_90.yaml", "w") as f:
        yaml.dump(cfg_b, f)

def evaluate_run(video_path, model_name, conf, tracker_cfg):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    model = YOLO(model_name)
    track_history = defaultdict(list)
    total_detections = 0
    pole_detections = 0  # detections in the pole region: x in [50, 160], y in [300, 550] with low displacement
    
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
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                score = float(boxes.conf[i].cpu().numpy())
                track_id = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else -1
                
                # Check if in pole region
                if 50 <= x1 <= 160 and 300 <= y1 <= 550 and (x2 - x1) < 120 and (y2 - y1) > 100:
                    pole_detections += 1
                    
                total_detections += 1
                if track_id >= 0:
                    track_history[track_id].append((frame_idx, [x1, y1, x2, y2], score))
                    
        frame_idx += 1
        
    cap.release()
    
    # Analyze tracks
    num_tracks = len(track_history)
    long_tracks = [t for t in track_history.values() if len(t) >= 30] # >= 1 second
    short_flicker_tracks = [t for t in track_history.values() if len(t) < 10]
    
    # Calculate moving tracks (> 100px displacement)
    moving_tracks = 0
    for t in track_history.values():
        if len(t) > 1:
            start_center = np.array([(t[0][1][0] + t[0][1][2])/2, (t[0][1][1] + t[0][1][3])/2])
            end_center = np.array([(t[-1][1][0] + t[-1][1][2])/2, (t[-1][1][1] + t[-1][1][3])/2])
            if np.linalg.norm(end_center - start_center) > 100:
                moving_tracks += 1
                
    result = {
        "model": model_name,
        "conf": conf,
        "tracker": os.path.basename(tracker_cfg),
        "total_detections": total_detections,
        "pole_detections": pole_detections,
        "unique_track_ids": num_tracks,
        "long_tracks": len(long_tracks),
        "short_flickers": len(short_flicker_tracks),
        "moving_tracks": moving_tracks,
    }
    return result

def main():
    create_tracker_configs()
    video_path = "videos/my_test.mp4"
    
    configs = [
        # Baseline
        ("yolov8n.pt", 0.35, "bytetrack.yaml"),
        
        # Test confidence levels with yolov8n
        ("yolov8n.pt", 0.45, "bytetrack.yaml"),
        ("yolov8n.pt", 0.50, "bytetrack.yaml"),
        
        # Test yolov8s with various confidences
        ("yolov8s.pt", 0.35, "bytetrack.yaml"),
        ("yolov8s.pt", 0.45, "bytetrack.yaml"),
        ("yolov8s.pt", 0.50, "bytetrack.yaml"),
        
        # Test tuned ByteTrack with yolov8s
        ("yolov8s.pt", 0.40, "models/trackers/bytetrack_tuned_60.yaml"),
        ("yolov8s.pt", 0.45, "models/trackers/bytetrack_tuned_60.yaml"),
        ("yolov8s.pt", 0.45, "models/trackers/bytetrack_tuned_90.yaml"),
    ]
    
    print("\n" + "="*80)
    print("RUNNING CONFIGURATION GRID EVALUATION ON videos/my_test.mp4")
    print("="*80)
    print(f"{'Model':<12} {'Conf':<6} {'Tracker':<26} {'Detections':<11} {'Pole FP':<9} {'Unique IDs':<11} {'Long Tracks':<12} {'Flickers':<9} {'Moving'}")
    print("-" * 105)
    
    for model_name, conf, tracker_cfg in configs:
        res = evaluate_run(video_path, model_name, conf, tracker_cfg)
        print(f"{res['model']:<12} {res['conf']:<6.2f} {res['tracker']:<26} {res['total_detections']:<11} {res['pole_detections']:<9} {res['unique_track_ids']:<11} {res['long_tracks']:<12} {res['short_flickers']:<9} {res['moving_tracks']}")

if __name__ == "__main__":
    main()
