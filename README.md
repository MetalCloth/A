# 🎥 YouTube Video Similarity Finder

An AI-powered tool that detects re-uploads, edited copies, and similar videos on YouTube using advanced computer vision and natural language processing.

## 📋 Overview

This application analyzes YouTube videos to identify potential duplicates, re-uploads, or edited versions by comparing visual content, titles, and metadata. It uses CLIP embeddings for visual similarity, text embeddings for semantic comparison, and an optional LLM-based classification agent for intelligent categorization.

## ✨ Features

- **🎞️ Keyframe Extraction**: Extracts evenly spaced frames from videos for analysis
- **🔍 YouTube Search**: Automatically searches for potential matches
- **🖼️ Thumbnail Filtering**: Quick pre-filtering using thumbnail similarity
- **🔬 Deep Frame Analysis**: Detailed frame-by-frame comparison using CLIP embeddings
- **🤖 AI Classification**: Optional Gemini-powered intelligent classification
- **📊 Multiple Similarity Metrics**: Visual, textual, and temporal comparisons
- **🎨 Interactive UI**: Clean Streamlit interface with progress tracking

https://github.com/user-attachments/assets/ca188a84-3607-4814-9ea6-27fbb643e6df


## 🏗️ Architecture

### Core Components

1. **Video Processor** (`video_processor.py`)
   - Metadata extraction using yt-dlp
   - Direct video stream processing with OpenCV
   - Evenly spaced keyframe extraction

2. **Embedding Models** (`embeddings.py`)
   - CLIP for visual embeddings
   - Sentence Transformers for text embeddings
   - Cached model loading for performance

3. **YouTube Search Service** (`youtube_search.py`)
   - YouTube search via yt-dlp
   - Thumbnail downloading with proper headers

4. **Similarity Service** (`similarity.py`)
   - Thumbnail-based filtering
   - Multi-metric similarity calculation
   - Cosine similarity comparisons

5. **Classification Agent** (`classification_agent.py`)
   - LangGraph workflow for structured classification
   - Gemini API integration
   - Rule-based fallback system

## 🚀 Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- Git

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd youtube-similarity-finder
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install system dependencies**

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html)

4. **Configure environment variables**

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

*(Optional: The app works without Gemini API, using rule-based classification)*

## 📦 Requirements

Create a `requirements.txt` file with:

```
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
yt-dlp>=2023.7.6
opencv-python>=4.8.0
Pillow>=10.0.0
requests>=2.31.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
langgraph>=0.0.20
langchain-google-genai>=1.0.0
google-generativeai>=0.3.0
numpy>=1.24.0
```

## 🎮 Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Enter a YouTube URL** in the input field
2. **Click "🔍 Scan"** to start the analysis
3. **Wait for processing** (typically 1-3 minutes depending on video length)
4. **Review results** showing:
   - Visual similarity percentage
   - Title similarity
   - Duration differences
   - AI classification (Re-upload/Edited Copy/Unrelated)
   - Confidence scores and reasoning

### Example

```
Input: https://www.youtube.com/watch?v=dQw4w9WgXcQ

Output:
✅ Found 5 potential matches
  #1 - Exact Re-upload (98.5% visual match)
  #2 - Edited Copy (87.3% visual match)
  #3 - Unrelated (45.2% visual match)
```

## ⚙️ Configuration

Edit `settings.py` to customize:

```python
# Model Configuration
clip_model_name = "openai/clip-vit-base-patch32"
text_embedding_model = "BAAI/bge-large-en-v1.5"
gemini_model = "gemini-2.5-flash"

# Processing Settings
num_keyframes = 7              # Frames to extract per video
max_youtube_results = 20        # Max search results
top_candidates = 5              # Videos to analyze deeply

# Classification Thresholds
reupload_threshold = 0.95       # Visual similarity for re-uploads
edited_copy_threshold = 0.85    # Visual similarity for edited copies
duration_diff_threshold = 2     # Max duration difference (seconds)
```

## 🔍 How It Works

### Step-by-Step Process

1. **Metadata Extraction**
   - Fetches title, duration, channel, thumbnail from source video

2. **Keyframe Extraction**
   - Extracts 7 evenly spaced frames using OpenCV
   - Resizes to 640x360 for consistency

3. **Visual Embedding Generation**
   - Generates CLIP embeddings for each frame
   - Averages embeddings into single representation

4. **YouTube Search**
   - Searches YouTube using video title
   - Returns up to 20 candidate videos

5. **Thumbnail Pre-filtering**
   - Downloads thumbnails for candidates
   - Computes similarity to source thumbnail
   - Selects top 5 most similar

6. **Deep Frame Analysis**
   - Extracts keyframes from each candidate
   - Generates CLIP embeddings
   - Calculates multiple similarity metrics

7. **AI Classification**
   - Uses Gemini to classify relationships
   - Provides confidence scores and reasoning
   - Falls back to rule-based if API unavailable

### Classification Categories

| Category | Visual Similarity | Duration Diff | Description |
|----------|------------------|---------------|-------------|
| **Re-upload** | >95% | <2s | Nearly identical content |
| **Edited Copy** | 85-95% | Any | Same content with modifications |
| **Unrelated** | <85% | Any | Different or coincidentally similar |

## 🛠️ Troubleshooting

### Common Issues

**1. "Failed to extract frames"**
- Ensure FFmpeg is installed and in PATH
- Try a different video (some may have restrictions)

**2. "OpenCV failed to open video stream"**
- Video may be age-restricted or geographically blocked
- Check internet connection

**3. "Gemini API error"**
- Verify API key in `.env` file
- Check API quota and billing
- App will use fallback classification if API fails

**4. Slow performance**
- Reduce `num_keyframes` in settings
- Reduce `max_youtube_results` or `top_candidates`
- Ensure GPU is available for PyTorch

### Performance Tips

- First run will be slower (model downloading)
- Models are cached after first load
- GPU acceleration significantly speeds up processing
- Consider using smaller CLIP models for faster inference

## 📊 Technical Details

### Similarity Metrics

**Visual Similarity**
- CLIP embeddings with cosine similarity
- Range: 0.0 (completely different) to 1.0 (identical)

**Title Similarity**
- BGE embeddings with cosine similarity
- Captures semantic meaning, not just word matching

**Duration Difference**
- Absolute difference in seconds
- Helps identify trimmed or extended versions

### Model Architecture

**CLIP (Contrastive Language-Image Pre-training)**
- Model: `openai/clip-vit-base-patch32`
- Jointly trained on image-text pairs
- Excellent for visual similarity

**BGE (BAAI General Embedding)**
- Model: `BAAI/bge-large-en-v1.5`
- State-of-the-art text embeddings
- Superior semantic understanding

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for batch processing
- [ ] Implement video timeline comparison
- [ ] Add audio fingerprinting
- [ ] Support for other video platforms
- [ ] Export results to CSV/JSON
- [ ] Add video preview player
- [ ] Implement caching for processed videos

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **OpenAI CLIP** - Visual embedding model
- **BAAI** - Text embedding model
- **yt-dlp** - YouTube downloading
- **Google Gemini** - AI classification
- **Streamlit** - Web interface framework

## ⚠️ Disclaimer

This tool is for research and educational purposes. Respect copyright laws and platform terms of service. Do not use for malicious purposes or copyright infringement detection without proper authorization.

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section above

---

**Built with ❤️ using AI and Computer Vision**
