import streamlit as st

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

    # -----------------------------
    # INITIALIZE SERVICES
    # -----------------------------
    @st.cache_resource
    def init_services():
        embedding_models = EmbeddingModels(
            clip_model_name=settings.clip_model_name,
            text_model_name=settings.text_embedding_model
        ).load_models()

        video_processor = VideoProcessor()
        youtube_service = YouTubeSearchService()
        similarity_service = SimilarityService(embedding_models)

        classification_agent = None
        if settings.gemini_api_key:
            classification_agent = ClassificationAgent(
                settings.gemini_api_key,
                settings.gemini_model
            )

        return {
            "embedding_models": embedding_models,
            "video_processor": video_processor,
            "youtube_service": youtube_service,
            "similarity_service": similarity_service,
            "classification_agent": classification_agent
        }

    with st.spinner("🔧 Loading AI models..."):
        services = init_services()

    # -----------------------------
    # INPUT
    # -----------------------------
    st.subheader("📝 Enter Video URL")
    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any YouTube video URL"
    )

    scan_button = st.button("🔍 Scan", type="primary", use_container_width=True)

    # -----------------------------
    # PIPELINE
    # -----------------------------
    if scan_button and video_url:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # -------------------------
            # STEP 1 — SOURCE METADATA
            # -------------------------
            status_text.text("📥 Step 1/6: Fetching source metadata...")
            progress_bar.progress(10)

            source_metadata = services["video_processor"].get_video_metadata(video_url)

            st.success(f"✅ Source: {source_metadata['title']}")

            # -------------------------
            # STEP 2 — SOURCE KEYFRAMES
            # -------------------------
            status_text.text("🎞️ Step 2/6: Extracting source keyframes...")
            progress_bar.progress(20)

            # 👇 ADD THIS BLOCK
            try:
                source_frames, duration = services["video_processor"].extract_keyframes(
                    video_url,
                    num_frames=settings.num_keyframes
                )
            except ValueError as e:
                st.error(f"❌ Failed to extract frames: {e}")
                st.stop()

            source_metadata["duration"] = duration

            # -------------------------
            # STEP 3 — SOURCE EMBEDDINGS
            # -------------------------
            status_text.text("🧠 Step 3/6: Generating visual embeddings...")
            progress_bar.progress(30)

            source_embedding = services["embedding_models"].generate_visual_embeddings(
                source_frames
            )

            # -------------------------
            # STEP 4 — YOUTUBE SEARCH
            # -------------------------
            status_text.text("🔎 Step 4/6: Searching YouTube...")
            progress_bar.progress(40)

            candidates = services["youtube_service"].search(
                source_metadata["title"],
                max_results=settings.max_youtube_results
            )

            st.info(f"📊 Found {len(candidates)} potential matches")

            # -------------------------
            # STEP 5 — THUMBNAIL FILTER
            # -------------------------
            status_text.text("🖼️ Step 5/6: Filtering by thumbnails...")
            progress_bar.progress(50)

            top_candidates = services["similarity_service"].filter_by_thumbnail(
                source_embedding,
                candidates,
                top_k=settings.top_candidates
            )

            st.info(f"✨ Selected top {len(top_candidates)} candidates")

            # -------------------------
            # STEP 6 — DEEP ANALYSIS
            # -------------------------
            status_text.text("🔬 Step 6/6: Deep frame analysis...")
            progress_bar.progress(60)

            analysis_results = []

            for candidate in top_candidates:
                try:
                    st.write(f"🔍 Analyzing: {candidate['title'][:50]}...")

                    metadata = services["video_processor"].get_video_metadata(
                        candidate["url"]
                    )

                    frames, duration = services["video_processor"].extract_keyframes(
                        candidate["url"],
                        num_frames=settings.num_keyframes
                    )

                    metadata["duration"] = duration

                    candidate_embedding = services["embedding_models"].generate_visual_embeddings(
                        frames
                    )

                    metrics = services["similarity_service"].calculate_match_metrics(
                        source_embedding,
                        source_metadata,
                        candidate_embedding,
                        metadata
                    )

                    analysis_results.append({
                        "url": candidate["url"],
                        "title": metadata["title"],
                        "channel": metadata["channel"],
                        "thumbnail": candidate["thumbnail"],
                        **metrics
                    })

                except Exception as e:
                    st.warning(f"⚠️ Skipped {candidate['title'][:30]}: {e}")

            progress_bar.progress(80)

            # -------------------------
            # CLASSIFICATION
            # -------------------------
            status_text.text("🤖 Classifying with AI...")

            if services["classification_agent"]:
                final_results = services["classification_agent"].classify(
                    source_metadata,
                    analysis_results
                )
            else:
                for result in analysis_results:
                    visual = result["visual_similarity"]
                    duration_diff = result["duration_diff"]

                    if visual > settings.reupload_threshold and duration_diff < settings.duration_diff_threshold:
                        result["classification"] = "Re-upload"
                        result["confidence"] = "High"
                        result["reasoning"] = "Nearly identical visual content and duration"
                    elif visual > settings.edited_copy_threshold:
                        result["classification"] = "Edited Copy"
                        result["confidence"] = "Medium"
                        result["reasoning"] = "High visual similarity with some differences"
                    else:
                        result["classification"] = "Unrelated"
                        result["confidence"] = "Low"
                        result["reasoning"] = "Visual similarity may be coincidental"

                final_results = analysis_results

            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            # -------------------------
            # DISPLAY RESULTS
            # -------------------------
            st.markdown("---")
            st.header("📊 Analysis Results")

            for idx, result in enumerate(final_results, 1):
                with st.expander(
                    f"#{idx} - {result['title'][:60]} ({result['classification']})",
                    expanded=True
                ):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(result["thumbnail"], use_container_width=True)

                    with col2:
                        st.metric("Visual Match", f"{result['visual_similarity']*100:.1f}%")
                        st.metric("Title Match", f"{result['title_similarity']*100:.1f}%")
                        st.metric("Duration Diff", f"{result['duration_diff']:.0f}s")

                        st.markdown(f"**Confidence:** {result['confidence']}")
                        st.markdown(f"**Reasoning:** {result['reasoning']}")
                        st.markdown(f"**Channel:** {result['channel']}")
                        st.markdown(f"[Open Video]({result['url']})")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

    # -----------------------------
    # SIDEBAR
    # -----------------------------
    with st.sidebar:
        st.header("ℹ️ How It Works")
        st.markdown("""
        1. 🎞️ Extract evenly spaced keyframes  
        2. 🔎 Search YouTube  
        3. 🖼️ Filter via thumbnails  
        4. 🔬 Deep frame similarity  
        5. 🤖 LLM classification  
        """)


if __name__ == "__main__":
    main()
