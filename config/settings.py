from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    clip_model_name: str = "openai/clip-vit-base-patch32"
    text_embedding_model: str = "BAAI/bge-large-en-v1.5"  
    gemini_model: str = "gemini-2.5-flash"  
    
    # Processing Settings
    num_keyframes: int = 7
    max_youtube_results: int = 20
    top_candidates: int = 5
    
    reupload_threshold: float = 0.95
    edited_copy_threshold: float = 0.85
    duration_diff_threshold: int = 2  
    
    thumbnail_download_timeout: int = 5
    video_download_timeout: int = 300
    
    class Config:
        env_file = ".env"

settings = Settings()
