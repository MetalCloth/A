import torch
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import streamlit as st

@st.cache_resource
def load_models(clip_name, text_name):
    # to get clip model
    clip_model = CLIPModel.from_pretrained(clip_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    text_model = SentenceTransformer(text_name)
    
    return {
        'clip_model': clip_model,
        'clip_processor': clip_processor,
        'text_model': text_model
    }

def get_visual_embedding(models, frames):
    images = [Image.fromarray(frame) for frame in frames]
    inputs = models['clip_processor'](images=images, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        features = models['clip_model'].get_image_features(**inputs)
    
    avg_embedding = torch.mean(features, dim=0)
    return avg_embedding


def compare_embeddings(models, embedding1, frames):
    embedding2 = get_visual_embedding(models, frames)
    similarity = util.cos_sim(embedding1, embedding2)
    return float(similarity[0][0])


def get_text_similarity(models, text1, text2):
    emb1 = models['text_model'].encode(text1, convert_to_tensor=True)
    emb2 = models['text_model'].encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)
    return float(similarity[0][0])