import torch
from typing import List, Dict
from models.embeddings import EmbeddingModels
from services.youtube_search import YouTubeSearchService

class SimilarityService:
    """Handles similarity comparisons and filtering"""
    
    def __init__(self, embedding_models: EmbeddingModels):
        self.embedding_models = embedding_models
        self.youtube_service = YouTubeSearchService()
    
    def filter_by_thumbnail(
        self,
        source_embedding: torch.Tensor,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """Filter candidates by comparing thumbnails"""
        results = []
        
        for candidate in candidates:
            thumbnail = self.youtube_service.download_thumbnail(candidate['thumbnail'])
            if thumbnail is None:
                continue
            
            # Generate thumbnail embedding
            inputs = self.embedding_models.clip_processor(
                images=thumbnail,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                thumb_embedding = self.embedding_models.clip_model.get_image_features(**inputs)
            
            # Calculate similarity
            similarity = self.embedding_models.calculate_visual_similarity(
                source_embedding,
                thumb_embedding
            )
            
            candidate['thumbnail_similarity'] = similarity
            results.append(candidate)
        
        # Sort and return top K
        results.sort(key=lambda x: x['thumbnail_similarity'], reverse=True)
        return results[:top_k]
    
    def calculate_match_metrics(
        self,
        source_embedding: torch.Tensor,
        source_metadata: Dict,
        candidate_embedding: torch.Tensor,
        candidate_metadata: Dict
    ) -> Dict:
        """Calculate all similarity metrics between source and candidate"""
        visual_similarity = self.embedding_models.calculate_visual_similarity(
            source_embedding,
            candidate_embedding
        )
        
        title_similarity = self.embedding_models.calculate_text_similarity(
            source_metadata['title'],
            candidate_metadata['title']
        )
        
        duration_diff = abs(source_metadata['duration'] - candidate_metadata['duration'])
        
        same_channel = source_metadata['channel'] == candidate_metadata['channel']
        
        return {
            'visual_similarity': visual_similarity,
            'title_similarity': title_similarity,
            'duration_diff': duration_diff,
            'same_channel': same_channel,
        }