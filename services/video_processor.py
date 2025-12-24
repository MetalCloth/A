import yt_dlp
import cv2
import numpy as np
import subprocess
import json
import os
from typing import List, Dict, Tuple
import warnings

# Silence logs
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
warnings.filterwarnings("ignore")

class VideoProcessor:
    """Handles video metadata and keyframe extraction (Direct URL Mode)"""

    @staticmethod
    def get_video_metadata(url: str) -> Dict:
        try:
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
        except Exception:
            return {"title": "Unknown", "duration": 0, "channel": "Unknown", "thumbnail": ""}

    @staticmethod
    def extract_keyframes(
        url: str,
        num_frames: int = 7,
        width: int = 640,  # Note: OpenCV resizes differently, but we handle it below
        height: int = 360
    ) -> Tuple[List[np.ndarray], float]:
        
        # 1. Get Direct Stream URL
        try:
            # -f 18 = MP4 360p (Best for OpenCV)
            # -g = Get URL only
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
        except Exception as e:
            raise ValueError(f"Failed to get stream URL: {e}")

        if not stream_url:
            raise ValueError("yt-dlp returned no URL. Video might be unavailable.")

        # 2. Open Stream with OpenCV
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            raise ValueError("OpenCV failed to open the video stream.")

        # 3. Read Frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback for streams where frame count is unknown
            total_frames = int(cap.get(cv2.CAP_PROP_FPS) * 10) # Assume 10s if unknown

        frames = []
        
        # Calculate indices
        indices = np.linspace(0, total_frames - 5, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize to target width/height
                frame = cv2.resize(frame, (width, height))
                # Convert BGR (OpenCV) to RGB (Pillow/CLIP)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError("No frames extracted from stream.")

        # Get duration from metadata to be safe
        meta = VideoProcessor.get_video_metadata(url)
        return frames, meta['duration']