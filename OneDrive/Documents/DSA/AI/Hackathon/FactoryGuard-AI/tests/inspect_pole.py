"""
Inspect the 57 pole detections with yolov8s to see if they correspond to an actual person walking near that location.
"""
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("videos/my_test.mp4")
frame_idx = 0
pole_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    res = model.track(frame, persist=True, tracker="models/trackers/bytetrack_tuned_90.yaml", conf=0.45, classes=[0], verbose=False)
    if res and res[0].boxes is not None:
        for i in range(len(res[0].boxes)):
            xyxy = res[0].boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            if 50 <= x1 <= 160 and 300 <= y1 <= 550 and (x2 - x1) < 120 and (y2 - y1) > 100:
                conf = float(res[0].boxes.conf[i].cpu().numpy())
                tid = int(res[0].boxes.id[i].cpu().numpy()) if res[0].boxes.id is not None else -1
                pole_frames.append((frame_idx, tid, conf, [round(c, 1) for c in xyxy]))
    frame_idx += 1
cap.release()

print(f"Total detections in region: {len(pole_frames)}")
if pole_frames:
    print(f"Frame span: {pole_frames[0][0]} to {pole_frames[-1][0]}")
    track_ids = set(p[1] for p in pole_frames)
    print(f"Track IDs associated: {track_ids}")
    for p in pole_frames[:10]:
        print(f"  Frame {p[0]:3d} | ID {p[1]:2d} | conf {p[2]:.3f} | bbox {p[3]}")
