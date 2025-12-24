import subprocess
import cv2
import time
import os

# --- CONFIG ---
TEST_URL = "https://www.youtube.com/watch?v=OETDZw-7qF0"

def debug_print(msg):
    print(f"🕵️ [DEBUG] {msg}")

def test_direct_url():
    debug_print(f"Testing URL: {TEST_URL}")

    # 1. GET DIRECT STREAM URL
    debug_print("Getting Direct URL from yt-dlp...")
    try:
        # -f 18 forces MP4 (video+audio), -g gets the URL
        cmd = ["yt-dlp", "-f", "18", "-g", TEST_URL]
        
        # We need to capture the output (the URL)
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode != 0:
            print(f"❌ yt-dlp failed: {result.stderr}")
            return

        stream_url = result.stdout.strip()
        debug_print(f"Got Stream URL: {stream_url[:50]}...") # Print first 50 chars
        
    except FileNotFoundError:
        print("❌ CRITICAL: yt-dlp not found in PATH.")
        return

    # 2. OPEN WITH OPENCV
    debug_print("Attempting to open with OpenCV...")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("❌ OpenCV failed to open the URL.")
        return

    # 3. READ FRAMES
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    debug_print(f"Video opened! Total frames: {total_frames}")

    frames_captured = 0
    # Try reading 5 frames
    for i in range(7):
        ret, frame = cap.read()
        if ret:
            frames_captured += 1
            print(f"   -> Captured Frame {i+1} ({frame.shape})")
        else:
            print(f"   -> Failed to read Frame {i+1}")
            break
            
    cap.release()
    
    if frames_captured > 0:
        print(f"\n✅ SUCCESS! Captured {frames_captured} frames.")
        print("   -> This method is MUCH safer. I will rewrite your VideoProcessor to use this.")
    else:
        print("\n❌ FAILED. OpenCV couldn't read the stream.")

if __name__ == "__main__":
    test_direct_url()