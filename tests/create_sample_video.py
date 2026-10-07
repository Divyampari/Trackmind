"""
Generate a test synthetic factory video for Phase 1 verification.
Creates a 3-second 30 FPS video with 2 simulated workers moving across a factory floor.
"""

import cv2
import numpy as np
import os

def create_test_video(output_path: str = "videos/test_factory.mp4", duration_sec: int = 3, fps: int = 30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for f in range(total_frames):
        # Factory floor background (gray grid)
        frame = np.full((height, width, 3), 60, dtype=np.uint8)
        
        # Grid lines
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (75, 75, 75), 1)
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (75, 75, 75), 1)
            
        # Draw worker 1 moving left to right
        w1_x = int(50 + (f / total_frames) * 400)
        w1_y = 200
        # Draw body
        cv2.circle(frame, (w1_x, w1_y), 18, (200, 180, 150), -1) # head
        cv2.rectangle(frame, (w1_x - 15, w1_y + 18), (w1_x + 15, w1_y + 80), (0, 100, 200), -1) # body (orange vest)
        cv2.line(frame, (w1_x - 8, w1_y + 80), (w1_x - 8, w1_y + 130), (50, 50, 50), 6) # left leg
        cv2.line(frame, (w1_x + 8, w1_y + 80), (w1_x + 8, w1_y + 130), (50, 50, 50), 6) # right leg
        
        # Draw worker 2 moving right to left
        w2_x = int(550 - (f / total_frames) * 350)
        w2_y = 220
        cv2.circle(frame, (w2_x, w2_y), 18, (210, 190, 160), -1)
        cv2.rectangle(frame, (w2_x - 15, w2_y + 18), (w2_x + 15, w2_y + 80), (50, 180, 50), -1)
        cv2.line(frame, (w2_x - 8, w2_y + 80), (w2_x - 8, w2_y + 130), (50, 50, 50), 6)
        cv2.line(frame, (w2_x + 8, w2_y + 80), (w2_x + 8, w2_y + 130), (50, 50, 50), 6)
        
        # Factory machinery/shelves
        cv2.rectangle(frame, (20, 20), (140, 120), (40, 40, 80), -1)
        cv2.putText(frame, "ZONE A", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        out.write(frame)
        
    out.release()
    print(f"Test video created: {output_path} ({total_frames} frames, {width}x{height} @ {fps}fps)")

if __name__ == "__main__":
    create_test_video()
