import yt_dlp
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple
import warnings

# FORCE SILENCE: Tell OpenCV/FFmpeg to shut up
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # Quiet
warnings.filterwarnings('ignore')

class VideoProcessor:
    """Handles video downloading and frame extraction"""
    
    @staticmethod
    def download_video(url: str, output_path: str) -> Tuple[str, Dict]:
        """Download video (Safe Mode)"""
        ydl_opts = {

            'format': '18/best[ext=mp4]',  
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        # Robust download
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Handle cases where thumbnail is a list or string
                thumb = info.get('thumbnail', '')
                if not thumb and info.get('thumbnails'):
                    thumb = info['thumbnails'][-1]['url']

                metadata = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'channel': info.get('uploader', info.get('channel', 'Unknown')),
                    'thumbnail': thumb,
                }
            return output_path, metadata
        except Exception as e:
            print(f"Download Error: {e}")
            raise e
    
    @staticmethod
    def extract_keyframes(video_path: str, num_frames: int = 7) -> Tuple[List[np.ndarray], float]:
        """Extract evenly spaced keyframes from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        frames = []
        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames, duration