"""
Detailed track verification with yolov8s.pt and bytetrack_tuned_90.yaml on my_test.mp4.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("videos/my_test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

track_history = defaultdict(list)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    res = model.track(frame, persist=True, tracker="models/trackers/bytetrack_tuned_90.yaml", conf=0.45, classes=[0], verbose=False)
    if res and res[0].boxes is not None:
        boxes = res[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            tid = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else -1
            if tid >= 0:
                track_history[tid].append((frame_idx, xyxy, conf))
    frame_idx += 1
cap.release()

print("\n" + "="*80)
print(f"TRACK PERSISTENCE REPORT (yolov8s.pt + bytetrack_tuned_90.yaml @ conf=0.45)")
print(f"Total Video Frames: {total_frames} | Total Track IDs: {len(track_history)}")
print("="*80)

for tid, frames in sorted(track_history.items(), key=lambda x: len(x[1]), reverse=True):
    start_f, end_f = frames[0][0], frames[-1][0]
    count = len(frames)
    span = end_f - start_f + 1
    continuity = (count / span) * 100
    avg_conf = np.mean([f[2] for f in frames])
    
    start_box = [round(float(c), 1) for c in frames[0][1]]
    end_box = [round(float(c), 1) for c in frames[-1][1]]
    
    start_c = np.array([(start_box[0]+start_box[2])/2, (start_box[1]+start_box[3])/2])
    end_c = np.array([(end_box[0]+end_box[2])/2, (end_box[1]+end_box[3])/2])
    displacement = np.linalg.norm(end_c - start_c)
    
    print(f"Track ID {tid:2d}: {count:4d} frames (span: {start_f:4d}-{end_f:4d}, {span:4d}f | {continuity:5.1f}% continuous) | "
          f"avg conf: {avg_conf:.3f} | disp: {displacement:6.1f}px | "
          f"start: {start_box} -> end: {end_box}")
