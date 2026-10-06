"""
Create a test video with realistic person images to verify YOLO person detection
and ByteTrack ID assignment across consecutive frames.
"""

import cv2
import numpy as np
import os
import urllib.request

def download_or_generate_person_image():
    """Download a standard COCO/sample image containing people or create one."""
    img_path = "data/sample_person.jpg"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    
    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    try:
        urllib.request.urlretrieve(url, img_path)
        img = cv2.imread(img_path)
        if img is not None:
            return img
    except Exception as e:
        print(f"Could not download sample image: {e}")
    
    # Fallback if offline: create blank canvas
    return np.zeros((720, 1280, 3), dtype=np.uint8)

def create_tracking_test_video(output_path: str = "videos/factory_workers_test.mp4", duration_sec: int = 2, fps: int = 30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_img = download_or_generate_person_image()
    
    h, w = base_img.shape[:2]
    # Resize to standard 720p if needed
    target_w, target_h = 1080, 720
    base_img = cv2.resize(base_img, (target_w, target_h))
    
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    
    for f in range(total_frames):
        # Slightly pan / jitter the image to simulate video frames
        shift_x = int(np.sin(f / 10.0) * 10)
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        frame = cv2.warpAffine(base_img, M, (target_w, target_h))
        out.write(frame)
        
    out.release()
    print(f"Created realistic test video: {output_path} ({total_frames} frames, {target_w}x{target_h} @ {fps}fps)")

if __name__ == "__main__":
    create_tracking_test_video()
