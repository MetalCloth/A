import streamlit as st
import torch
import numpy as np
from config.settings import settings
from services.video_processor import get_video_info, extract_frames
from services.youtube_search import search_youtube, download_thumbnail
from models.embeddings import load_models, get_visual_embedding, get_text_similarity, compare_embeddings
from apps.classification_agent import classify_with_ai

st.set_page_config(page_title="Video Similarity Finder", page_icon="🎥", layout="wide")

st.title("🎥 YouTube Video Similarity Finder")
st.markdown("Find re-uploads and edited copies of videos")

@st.cache_resource
def init_models():
    return load_models(settings.clip_model_name, settings.text_model_name)

with st.spinner("Loading AI models..."):
    models = init_models()

st.subheader(" Enter Video URL")
video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button(" Scan", type="primary", use_container_width=True):
    if not video_url:
        st.error("Please enter a video URL")
        st.stop()
    
    try:
        progress = st.progress(0)
        status = st.empty()
        
        status.text(" Getting source video info...")
        progress.progress(10)
        source_info = get_video_info(video_url)
        st.success(f" Source: {source_info['title']}")
        
        status.text(" Extracting frames from source...")
        progress.progress(20)
        try:
            source_frames = extract_frames(video_url, settings.num_keyframes)
        except Exception as e:
            st.error(f"Failed to extract frames: {e}")
            st.stop()
        
        status.text(" Creating visual fingerprint...")
        progress.progress(30)
        source_embedding = get_visual_embedding(models, source_frames)
        
        status.text(" Searching YouTube...")
        progress.progress(40)
        candidates = search_youtube(source_info['title'], settings.max_youtube_results)
        st.info(f" Found {len(candidates)} potential matches")
        
        status.text(" Filtering by thumbnails...")
        progress.progress(50)
        
        scored_candidates = []
        for candidate in candidates:
            thumb = download_thumbnail(candidate['thumbnail'])
            if thumb:
                thumb_score = compare_embeddings(models, source_embedding, [np.array(thumb)])
                scored_candidates.append({**candidate, 'score': thumb_score})
        
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = scored_candidates[:settings.top_candidates]
        st.info(f" Analyzing top {len(top_candidates)} candidates")
        
        status.text(" Deep frame analysis...")
        progress.progress(60)
        
        results = []
        for i, candidate in enumerate(top_candidates):
            st.write(f" Analyzing: {candidate['title'][:50]}...")
            
            try:
                cand_info = get_video_info(candidate['url'])
                cand_frames = extract_frames(candidate['url'], settings.num_keyframes)
                
                visual_match = compare_embeddings(models, source_embedding, cand_frames)
                title_match = get_text_similarity(models, source_info['title'], cand_info['title'])
                duration_diff = abs(source_info['duration'] - cand_info['duration'])
                same_channel = source_info['channel'] == cand_info['channel']
                
                results.append({
                    'url': candidate['url'],
                    'title': cand_info['title'],
                    'channel': cand_info['channel'],
                    'thumbnail': candidate['thumbnail'],
                    'visual_match': visual_match,
                    'title_match': title_match,
                    'duration_diff': duration_diff,
                    'same_channel': same_channel
                })
            except Exception as e:
                st.warning(f" Skipped {candidate['title'][:30]}: {e}")
        
        progress.progress(80)
        status.text(" AI classification...")
        
        if settings.gemini_api_key:
            final_results = classify_with_ai(source_info, results)
        else:
            for r in results:
                if r['visual_match'] > 0.95 and r['duration_diff'] < 2:
                    r['label'] = "Re-upload"
                    r['confidence'] = "High"
                    r['reason'] = "Nearly identical content"
                elif r['visual_match'] > 0.85:
                    r['label'] = "Edited Copy"
                    r['confidence'] = "Medium"
                    r['reason'] = "High visual similarity with differences"
                else:
                    r['label'] = "Unrelated"
                    r['confidence'] = "Low"
                    r['reason'] = "Low similarity"
            final_results = results
        
        progress.progress(100)
        status.text(" Done!")
        
        st.markdown("---")
        st.header(" Results")
        
        for idx, r in enumerate(final_results, 1):
            with st.expander(f"#{idx} - {r['title'][:60]} ({r['label']})", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(r['thumbnail'], use_container_width=True)
                
                with col2:
                    st.metric("Visual Match", f"{r['visual_match']*100:.1f}%")
                    st.metric("Title Match", f"{r['title_match']*100:.1f}%")
                    st.metric("Duration Diff", f"{r['duration_diff']:.0f}s")
                    st.markdown(f"**{r['label']}** ({r['confidence']} confidence)")
                    st.markdown(f"{r['reason']}")
                    st.markdown(f"Channel: {r['channel']}")
                    st.markdown(f"[Open Video]({r['url']})")
    
    except Exception as e:
        st.error(f" Error: {e}")
        st.exception(e)

with st.sidebar:
    st.header("How It Works")
    st.markdown("""
    1. Extract frames from your video
    2. Search YouTube for similar titles
    3. Filter by thumbnail similarity
    4. Deep analysis of top matches
    5. AI classification of results
    """)