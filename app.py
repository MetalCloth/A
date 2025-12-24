import streamlit as st
import tempfile
import os
from pathlib import Path

from config.settings import settings
from models.embeddings import EmbeddingModels
from services.video_processor import VideoProcessor
from services.youtube_search import YouTubeSearchService
from services.similarity import SimilarityService
from apps.classification_agent import ClassificationAgent
def main():
    st.set_page_config(
        page_title="Video Similarity Finder",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 YouTube Video Similarity Finder")
    st.markdown("AI-powered detection of re-uploads, edited copies, and similar videos")
    
    # Initialize services
    @st.cache_resource
    def init_services():
        embedding_models = EmbeddingModels(
            clip_model_name=settings.clip_model_name,
            text_model_name=settings.text_embedding_model
        ).load_models()
        
        st.write("VIDEO PROCESSOR LOADED")

        video_processor = VideoProcessor()

        st.write("Youtube search service loaded")

        youtube_service = YouTubeSearchService()


        st.write("Similarity service loaded")
        similarity_service = SimilarityService(embedding_models)
        
        classification_agent = None
        if settings.gemini_api_key:
            classification_agent = ClassificationAgent(
                settings.gemini_api_key,
                settings.gemini_model
            )
        
        return {
            'embedding_models': embedding_models,
            'video_processor': video_processor,
            'youtube_service': youtube_service,
            'similarity_service': similarity_service,
            'classification_agent': classification_agent
        }
    
    with st.spinner("🔧 Loading AI models..."):
        services = init_services()
        # st.write(services)
        st.write("Loaded Services:")
        for name in services:
            st.write(f"✅ {name}")
    
    # Input section
    st.subheader("📝 Enter Video URL")
    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=... or https://youtube.com/shorts/...",
        help="Paste any YouTube video URL"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        scan_button = st.button("🔍 Scan", type="primary", use_container_width=True)
    
    if scan_button and video_url:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Download source video
            status_text.text("📥 Step 1/6: Processing source video...")
            progress_bar.progress(10)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                source_path = os.path.join(tmpdir, "source.mp4")

                st.write("DOING THIS DOWNLOAD VIDEO STUDD I HATE YOU??")
                source_path, source_metadata = services['video_processor'].download_video(
                    video_url,
                    source_path
                )
                
                
                st.success(f"✅ Source: {source_metadata['title']}")
                
                # Step 2: Extract keyframes
                status_text.text("🎞️ Step 2/6: Extracting keyframes...")
                progress_bar.progress(20)
                
                source_frames, duration = services['video_processor'].extract_keyframes(
                    source_path,
                    num_frames=settings.num_keyframes
                )
                source_metadata['duration'] = duration
                
                # Step 3: Generate embeddings
                status_text.text("🧠 Step 3/6: Generating visual embeddings...")
                progress_bar.progress(30)
                
                source_embedding = services['embedding_models'].generate_visual_embeddings(
                    source_frames
                )
                
                # Step 4: Search YouTube
                status_text.text("🔎 Step 4/6: Searching YouTube...")
                progress_bar.progress(40)
                
                candidates = services['youtube_service'].search(
                    source_metadata['title'],
                    max_results=settings.max_youtube_results
                )
                
                st.info(f"📊 Found {len(candidates)} potential matches")
                
                # Step 5: Filter by thumbnails
                status_text.text("🖼️ Step 5/6: Filtering by thumbnails...")
                progress_bar.progress(50)
                
                top_candidates = services['similarity_service'].filter_by_thumbnail(
                    source_embedding,
                    candidates,
                    top_k=settings.top_candidates
                )
                
                st.info(f"✨ Selected top {len(top_candidates)} candidates for deep analysis")
                
                # Step 6: Deep analysis
                status_text.text("🔬 Step 6/6: Deep frame analysis...")
                progress_bar.progress(60)
                
                analysis_results = []
                for idx, candidate in enumerate(top_candidates):
                    try:
                        st.write(f"🔍 Analyzing: {candidate['title'][:50]}...")
                        
                        video_path = os.path.join(tmpdir, f"candidate_{idx}.mp4")
                        video_path, metadata = services['video_processor'].download_video(
                            candidate['url'],
                            video_path
                        )
                        
                        frames, duration = services['video_processor'].extract_keyframes(
                            video_path,
                            num_frames=settings.num_keyframes
                        )
                        
                        candidate_embedding = services['embedding_models'].generate_visual_embeddings(frames)
                        
                        metrics = services['similarity_service'].calculate_match_metrics(
                            source_embedding,
                            source_metadata,
                            candidate_embedding,
                            metadata
                        )
                        
                        analysis_results.append({
                            'url': candidate['url'],
                            'title': metadata['title'],
                            'channel': metadata['channel'],
                            'thumbnail': candidate['thumbnail'],
                            **metrics
                        })
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not analyze {candidate['title'][:30]}: {str(e)}")
                        continue
                
                progress_bar.progress(80)
                
                # LLM Classification
                status_text.text("🤖 Classifying with AI...")
                
                if services['classification_agent']:
                    final_results = services['classification_agent'].classify(
                        source_metadata,
                        analysis_results
                    )
                else:
                    st.warning("⚠️ GEMINI_API_KEY not set. Using fallback classification.")
                    # Use fallback
                    for result in analysis_results:
                        visual = result['visual_similarity']
                        duration = result['duration_diff']
                        
                        if visual > settings.reupload_threshold and duration < settings.duration_diff_threshold:
                            result['classification'] = 'Re-upload'
                            result['confidence'] = 'High'
                            result['reasoning'] = 'Nearly identical visual content and duration'
                        elif visual > settings.edited_copy_threshold:
                            result['classification'] = 'Edited Copy'
                            result['confidence'] = 'Medium'
                            result['reasoning'] = 'High visual similarity with some differences'
                        else:
                            result['classification'] = 'Unrelated'
                            result['confidence'] = 'Low'
                            result['reasoning'] = 'Visual similarity may be coincidental'
                    
                    final_results = analysis_results
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                for idx, result in enumerate(final_results, 1):
                    with st.expander(
                        f"#{idx} - {result['title'][:60]}... ({result.get('classification', 'Unknown')})",
                        expanded=True
                    ):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(result['thumbnail'], use_container_width=True)
                        
                        with col2:
                            classification = result.get('classification', 'Unknown')
                            if classification == 'Re-upload':
                                st.error(f"🚨 {classification}")
                            elif classification == 'Edited Copy':
                                st.warning(f"⚠️ {classification}")
                            else:
                                st.info(f"ℹ️ {classification}")
                            
                            st.markdown(f"**Confidence:** {result.get('confidence', 'N/A')}")
                            st.markdown(f"**Reasoning:** {result.get('reasoning', 'N/A')}")
                            st.markdown("---")
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Visual Match", f"{result['visual_similarity']*100:.1f}%")
                            with metric_col2:
                                st.metric("Title Match", f"{result['title_similarity']*100:.1f}%")
                            with metric_col3:
                                st.metric("Duration Diff", f"{result['duration_diff']:.0f}s")
                            
                            st.markdown(f"**Channel:** {result['channel']}")
                            st.markdown(f"**URL:** [{result['url']}]({result['url']})")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ How It Works")
        st.markdown("""
        **LangGraph Agent Pipeline:**
        
        1. 🎞️ **Extract Keyframes** - 7 frames from source
        2. 🔎 **YouTube Search** - Find 20 candidates
        3. 🖼️ **Thumbnail Filter** - Select top 5
        4. 🔬 **Deep Analysis** - Frame comparison
        5. 🤖 **AI Agent** - LangGraph classification
        6. 📊 **Results** - Categorized matches
        
        **Classifications:**
        - 🚨 **Re-upload** - Exact copy
        - ⚠️ **Edited Copy** - Modified
        - ℹ️ **Unrelated** - Different
        """)
        
        st.markdown("---")
        st.markdown("**⚙️ Configuration:**")
        st.json({
            "Keyframes": settings.num_keyframes,
            "Search Results": settings.max_youtube_results,
            "Top Candidates": settings.top_candidates,
            "Text Model": settings.text_embedding_model.split("/")[-1]
        })

if __name__ == "__main__":
    main()