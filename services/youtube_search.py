import yt_dlp
import requests
from PIL import Image
from io import BytesIO

def search_youtube(query, max_results=20):
    search_url = f"ytsearch{max_results}:{query}"
    opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }
    
    with yt_dlp.YoutubeDL(opts) as ydl:
        results = ydl.extract_info(search_url, download=False)
    
    candidates = []
    if 'entries' in results:
        for entry in results['entries']:
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

def download_thumbnail(url, timeout=5):
    if not url:
        return None
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, timeout=timeout, headers=headers)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img.convert('RGB')
    except:
        pass
    
    return None