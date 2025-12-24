import numpy as np
import subprocess
import torch
from transformers import ClapModel, ClapProcessor
import tempfile
import os

# --- CONFIG ---
TEST_URL = "https://www.youtube.com/watch?v=OETDZw-7qF0"
FFMPEG_BINARY = r"C:\Users\rawat\anaconda3\envs\yt_something\Library\bin\ffmpeg.exe"
MODEL_NAME = "laion/clap-htsat-unfused"


def download_and_extract_audio(url):
    """Download audio, then extract 5 seconds"""
    print("📥 Downloading audio...")
    
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        tmp_path = tmp.name
    
    print(f"   Temp file: {tmp_path}")
    
    try:
        # Download with yt-dlp (show output for debugging)
        dl_cmd = [
    "yt-dlp",
    "-f", "bestaudio",
    "--extractor-args", "youtube:player_client=android",  # Use Android client (no JS needed)
    "-o", tmp_path,
    url
]
        print(f"   Running: {' '.join(dl_cmd[:3])}...")
        
        result = subprocess.run(dl_cmd, capture_output=True, timeout=120, text=True)
        
        # Print yt-dlp output
        print(f"\n--- yt-dlp stdout ---")
        print(result.stdout[:500] if result.stdout else "(empty)")
        print(f"\n--- yt-dlp stderr ---")
        print(result.stderr[:500] if result.stderr else "(empty)")
        print(f"--- Return code: {result.returncode} ---\n")
        
        if result.returncode != 0:
            print(f"❌ Download failed")
            return None
        
        # Check file
        if os.path.exists(tmp_path):
            size = os.path.getsize(tmp_path)
            print(f"✅ File exists: {size:,} bytes")
            if size == 0:
                print("❌ But file is empty!")
                return None
        else:
            print(f"❌ File doesn't exist at: {tmp_path}")
            return None
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def get_ai_fingerprint(audio_array):
    """Generate AI embedding from audio"""
    print("🤖 Loading AI model...")
    
    model = ClapModel.from_pretrained(MODEL_NAME)
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    
    print("🤖 Processing audio...")
    inputs = processor(
        audios=audio_array, 
        sampling_rate=48000, 
        return_tensors="pt", 
        padding=True
    )
    
    with torch.no_grad():
        embedding = model.get_audio_features(**inputs)
        
    print(f"✅ Generated {embedding.shape[1]}-dimensional fingerprint")
    return embedding[0]

# --- MAIN ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎵 AUDIO FINGERPRINTING SYSTEM 🎵")
    print("="*50 + "\n")
    
    # Download and extract audio
    audio_data = download_and_extract_audio(TEST_URL)
    if audio_data is None:
        print("\n❌ FAILED: Could not get audio")
        exit(1)
    
    # Generate fingerprint
    vector = get_ai_fingerprint(audio_data)
    
    print("\n" + "="*50)
    print("🔍 FINGERPRINT RESULTS")
    print("="*50)
    print(f"First 10 values: {vector[:10].numpy()}")
    print(f"Vector shape: {vector.shape}")
    print(f"Vector norm: {torch.norm(vector).item():.4f}")
    print("\n✅ SUCCESS!\n")