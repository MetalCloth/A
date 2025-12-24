import subprocess
import json
import cv2
import numpy as np
import warnings
import os

# os.environ["OPENCV_LOG_LEVEL"] = "OFF"
# warnings.filterwarnings("ignore")

def get_video_info(url):
    result = subprocess.run(
        ["yt-dlp", "-J", "--no-check-certificate", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    
    info = json.loads(result.stdout)
    thumb = info.get("thumbnail", "")
    if not thumb and info.get("thumbnails"):
        thumb = info["thumbnails"][-1]["url"]
    
    return {
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0.0),
        "channel": info.get("uploader", info.get("channel", "Unknown")),
        "thumbnail": thumb
    }

def extract_frames(url, num_frames=7, width=640, height=360):
    cmd = ["yt-dlp", "-f", "18/best[ext=mp4]", "-g", "--no-check-certificate", url]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    
    stream_url = result.stdout.strip()
    if not stream_url:
        raise ValueError("Could not get video stream URL")
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ValueError("Failed to open video stream")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(fps * 10) if fps > 0 else 300
    
    frames = []
    indices = np.linspace(0, max(total_frames - 5, 1), num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames could be extracted")
    
    return frames