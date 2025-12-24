import torch
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List
import streamlit as st

class EmbeddingModels:
    """Manages all embedding models"""
    
    def __init__(self, clip_model_name: str, text_model_name: str):
        self.clip_model = None
        self.clip_processor = None
        self.text_model = None
        self.clip_model_name = clip_model_name
        self.text_model_name = text_model_name
    
    @st.cache_resource
    def load_models(_self):
        """Load and cache models"""
        _self.clip_model = CLIPModel.from_pretrained(_self.clip_model_name)
        _self.clip_processor = CLIPProcessor.from_pretrained(_self.clip_model_name)
        _self.text_model = SentenceTransformer(_self.text_model_name)
        return _self
    
    def generate_visual_embeddings(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Generate CLIP embeddings for video frames"""
        images = [Image.fromarray(frame) for frame in frames]
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        avg_embedding = torch.mean(image_features, dim=0)
        return avg_embedding
    
    
    def generate_text_embedding(self, text: str) -> torch.Tensor:
        """Generate text embedding"""
        return self.text_model.encode(text, convert_to_tensor=True)
    
    def calculate_visual_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculate cosine similarity between visual embeddings"""
        cos_sim = util.cos_sim(emb1, emb2)
        return float(cos_sim[0][0])
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        emb1 = self.generate_text_embedding(text1)
        emb2 = self.generate_text_embedding(text2)
        cos_sim = util.cos_sim(emb1, emb2)
        return float(cos_sim[0][0])