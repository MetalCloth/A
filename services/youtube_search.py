import yt_dlp
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Optional

class YouTubeSearchService:
    """Handles YouTube search operations"""
    
    @staticmethod
    def search(query: str, max_results: int = 20) -> List[Dict]:
        """Search YouTube and return candidate videos"""
        search_url = f"ytsearch{max_results}:{query}"
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True, # Fast search
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(search_url, download=False)
        
        candidates = []
        if 'entries' in results:
            for entry in results['entries']:
                # FIX: Handle missing thumbnails safer
                thumb_url = entry.get('thumbnail', '')
                if not thumb_url and entry.get('thumbnails'):
                     thumb_url = entry['thumbnails'][0]['url']
                
                candidates.append({
                    'video_id': entry.get('id', ''),
                    'title': entry.get('title', ''),
                    'url': f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                    'thumbnail': thumb_url,
                    'duration': entry.get('duration', 0),
                    'channel': entry.get('uploader', 'Unknown'),
                })
        
        return candidates
    
    @staticmethod
    def download_thumbnail(url: str, timeout: int = 5) -> Optional[Image.Image]:
        """Download thumbnail image with Bot Headers"""
        if not url:
            return None
            
        # FIX: Add User-Agent so YouTube doesn't block the image download
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return img.convert('RGB') # Fix for some PNGs
            return None
        except Exception as e:
            # print(f"Thumbnail fail: {e}") # Debug only
            return None