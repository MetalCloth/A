import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    clip_model_name = "openai/clip-vit-base-patch32"
    text_model_name = "BAAI/bge-large-en-v1.5"
    gemini_model = "gemini-2.5-flash"
    
    num_keyframes = 7
    max_youtube_results = 20
    top_candidates = 5
    
    reupload_threshold = 0.95
    edited_copy_threshold = 0.85
    duration_diff_threshold = 2

settings = Settings()