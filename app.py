"""
AI-Powered Video Analyzer
A Streamlit app that analyzes videos and generates executive summaries with actionable insights.
Uses OpenRouter API for LLM access and OpenAI Whisper for transcription.
"""

import os
import tempfile
import shutil
import subprocess
import re

import requests
import streamlit as st

from src.video_processor import process_video, format_duration
from src.transcription import transcribe_audio, format_transcript_with_timestamps
from src.visual_analyzer import analyze_frames, summarize_visual_content, VISION_MODELS
from src.analyzer import generate_analysis, generate_topic_tags, ANALYSIS_MODELS


# Available models
MODELS = {
    "Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "GPT-5 Mini": "openai/gpt-5-mini",
    "Gemini 3 Flash": "google/gemini-3-flash-preview",
}

# Page configuration
st.set_page_config(
    page_title="AI Video Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing results
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .tag {
        display: inline-block;
        background-color: #e9ecef;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_keys() -> tuple[bool, list]:
    """Check if required API keys are configured."""
    missing = []

    if not st.secrets.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY (for Whisper transcription)")

    if not st.secrets.get("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY (for AI analysis)")

    return len(missing) == 0, missing


def convert_google_drive_url(url: str) -> str:
    """Convert Google Drive sharing URL to direct download URL."""
    patterns = [
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"

    return url


def convert_dropbox_url(url: str) -> str:
    """Convert Dropbox sharing URL to direct download URL."""
    if "dropbox.com" in url:
        url = re.sub(r'dl=0', 'dl=1', url)
        if 'dl=1' not in url:
            url = url + ('&' if '?' in url else '?') + 'dl=1'
    return url


def download_video_from_url(url: str, output_dir: str, progress_callback=None) -> str:
    """Download video from URL to local file."""
    if "drive.google.com" in url:
        url = convert_google_drive_url(url)
    elif "dropbox.com" in url:
        url = convert_dropbox_url(url)

    filename = "video.mp4"
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    content_disp = response.headers.get('content-disposition', '')
    if 'filename=' in content_disp:
        match = re.search(r'filename="?([^";\n]+)"?', content_disp)
        if match:
            filename = match.group(1)

    total_size = int(response.headers.get('content-length', 0))
    output_path = os.path.join(output_dir, filename)

    downloaded = 0
    chunk_size = 8192 * 1024

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    progress_callback(downloaded, total_size)

    return output_path


def run_analysis(video_path: str, video_name: str, temp_dir: str,
                 analyze_visuals: bool, num_frames: int,
                 vision_model: str, vision_model_name: str,
                 analysis_model: str, analysis_model_name: str,
                 custom_instructions: str) -> dict:
    """Run the full analysis pipeline on a video. Returns results dict."""

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Process video
    status_text.text("Step 1/4: Processing video...")
    progress_bar.progress(10)

    video_data = process_video(
        video_path,
        output_dir=temp_dir,
        extract_frames=analyze_visuals,
        num_frames=num_frames
    )

    st.success(f"Video duration: {video_data['duration_formatted']}")

    # Step 2: Transcribe audio
    status_text.text("Step 2/4: Transcribing audio (this may take a few minutes)...")
    progress_bar.progress(25)

    transcript_data = transcribe_audio(video_data["audio_path"])
    transcript = transcript_data["text"]

    st.success(f"Transcription complete ({len(transcript):,} characters)")

    # Step 3: Analyze visual content (optional)
    visual_summary = None
    if analyze_visuals and video_data["frames"]:
        status_text.text(f"Step 3/4: Analyzing visual content with {vision_model_name}...")

        def update_frame_progress(current, total):
            progress = 40 + int((current / total) * 20)
            progress_bar.progress(progress)

        frame_analyses = analyze_frames(
            video_data["frames"],
            model=vision_model,
            progress_callback=update_frame_progress
        )

        visual_summary = summarize_visual_content(
            frame_analyses,
            model=vision_model
        )
        st.success(f"Analyzed {len(frame_analyses)} key frames")
    else:
        progress_bar.progress(60)

    # Step 4: Generate comprehensive analysis
    status_text.text(f"Step 4/4: Generating analysis with {analysis_model_name}...")
    progress_bar.progress(70)

    analysis = generate_analysis(
        transcript=transcript,
        visual_summary=visual_summary,
        video_duration=video_data["duration_formatted"],
        custom_instructions=custom_instructions if custom_instructions else None,
        model=analysis_model
    )

    progress_bar.progress(90)

    # Generate topic tags
    tags = generate_topic_tags(transcript, model=analysis_model)

    progress_bar.progress(100)
    status_text.text("Analysis complete!")

    # Return all results as a dictionary to store in session state
    return {
        "analysis": analysis,
        "transcript": transcript,
        "transcript_data": transcript_data,
        "visual_summary": visual_summary,
        "video_data": {
            "duration_formatted": video_data["duration_formatted"]
        },
        "video_name": video_name,
        "tags": tags,
        "analysis_model_name": analysis_model_name,
        "vision_model_name": vision_model_name
    }


def display_results(results: dict):
    """Display the analysis results from session state."""

    analysis = results["analysis"]
    transcript = results["transcript"]
    transcript_data = results["transcript_data"]
    visual_summary = results["visual_summary"]
    video_data = results["video_data"]
    video_name = results["video_name"]
    tags = results["tags"]
    analysis_model_name = results["analysis_model_name"]
    vision_model_name = results["vision_model_name"]

    st.divider()

    # Topic tags
    if tags:
        st.markdown("**Topics:** " + " ".join([f"`{tag}`" for tag in tags]))

    st.divider()

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Executive Summary",
        "Key Insights & Actions",
        "Full Analysis",
        "Transcript"
    ])

    with tab1:
        if "Executive Summary" in analysis["sections"]:
            st.markdown(analysis["sections"]["Executive Summary"])

        if "Content Overview" in analysis["sections"]:
            st.markdown("### Content Overview")
            st.markdown(analysis["sections"]["Content Overview"])

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            if "Key Insights" in analysis["sections"]:
                st.markdown("### Key Insights")
                st.markdown(analysis["sections"]["Key Insights"])

            if "Notable Quotes" in analysis["sections"]:
                st.markdown("### Notable Quotes")
                st.markdown(analysis["sections"]["Notable Quotes"])

        with col2:
            if "Actionable Takeaways" in analysis["sections"]:
                st.markdown("### Actionable Takeaways")
                st.markdown(analysis["sections"]["Actionable Takeaways"])

            if "Questions Raised" in analysis["sections"]:
                st.markdown("### Questions Raised")
                st.markdown(analysis["sections"]["Questions Raised"])

    with tab3:
        st.markdown(analysis["full_analysis"])

        if visual_summary:
            st.markdown("---")
            st.markdown("### Visual Content Summary")
            st.markdown(visual_summary)

    with tab4:
        st.markdown("### Full Transcript")

        if "segments" in transcript_data and transcript_data["segments"]:
            formatted = format_transcript_with_timestamps(transcript_data["segments"])
            st.text_area(
                "Transcript with timestamps",
                formatted,
                height=400,
                label_visibility="collapsed"
            )
        else:
            st.text_area(
                "Transcript",
                transcript,
                height=400,
                label_visibility="collapsed"
            )

    # Download options
    st.divider()
    st.markdown("### Download Results")

    # Prepare download content
    full_report = f"""# Video Analysis Report

**Video:** {video_name}
**Duration:** {video_data['duration_formatted']}
**Topics:** {', '.join(tags)}
**Analysis Model:** {analysis_model_name}
**Vision Model:** {vision_model_name}

---

{analysis['full_analysis']}

---

## Visual Content Summary

{visual_summary if visual_summary else 'Visual analysis not performed.'}

---

## Full Transcript

{transcript}
"""

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Download Full Analysis",
            analysis["full_analysis"],
            file_name=f"{video_name}_analysis.md",
            mime="text/markdown",
            key="download_analysis"
        )

    with col2:
        st.download_button(
            "Download Transcript",
            transcript,
            file_name=f"{video_name}_transcript.txt",
            mime="text/plain",
            key="download_transcript"
        )

    with col3:
        st.download_button(
            "Download Complete Report",
            full_report,
            file_name=f"{video_name}_complete_report.md",
            mime="text/markdown",
            key="download_complete"
        )

    # Button to clear results and analyze another video
    st.divider()
    if st.button("Analyze Another Video", type="secondary"):
        st.session_state.analysis_results = None
        st.rerun()


def main():
    # Header
    st.markdown('<p class="main-header">AI-Powered Video Analyzer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a video or provide a URL to get an executive summary with actionable insights</p>',
        unsafe_allow_html=True
    )

    # Check API keys
    keys_ok, missing_keys = check_api_keys()

    if not keys_ok:
        st.error(f"Missing API keys:")
        for key in missing_keys:
            st.markdown(f"- {key}")
        st.info("Please add your API keys in Streamlit's Secrets management (Settings > Secrets)")
        st.code("""
# Add these to your Streamlit secrets:
OPENAI_API_KEY = "sk-..."           # For Whisper transcription
OPENROUTER_API_KEY = "sk-or-..."    # For AI analysis (OpenRouter)
        """)
        return

    # If we have results in session state, display them
    if st.session_state.analysis_results is not None:
        st.success("Analysis complete! Results are shown below.")
        display_results(st.session_state.analysis_results)
        return

    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")

        st.subheader("AI Models")

        analysis_model_name = st.selectbox(
            "Analysis Model",
            options=list(MODELS.keys()),
            index=0,
            help="Model used for generating summaries and insights"
        )
        analysis_model = MODELS[analysis_model_name]

        vision_model_name = st.selectbox(
            "Vision Model",
            options=list(MODELS.keys()),
            index=2,
            help="Model used for analyzing video frames"
        )
        vision_model = MODELS[vision_model_name]

        st.divider()

        st.subheader("Visual Analysis")

        analyze_visuals = st.checkbox(
            "Analyze visual content",
            value=True,
            help="Extract and analyze key frames from the video"
        )

        num_frames = st.slider(
            "Number of frames to analyze",
            min_value=5,
            max_value=20,
            value=10,
            disabled=not analyze_visuals,
            help="More frames = more visual context but higher cost"
        )

        st.divider()

        custom_instructions = st.text_area(
            "Custom analysis instructions (optional)",
            placeholder="E.g., 'Focus on marketing strategies mentioned' or 'Highlight technical implementation details'",
            help="Add specific instructions to guide the analysis"
        )

        st.divider()

        st.markdown("### Estimated Costs")
        st.markdown("""
        - **Whisper API**: ~$0.006/min
        - **OpenRouter**: Varies by model

        **~$0.50-1.00 per hour of video**
        """)

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload file (up to 200MB)", "Video URL (for larger files)"],
        horizontal=True
    )

    if input_method == "Upload file (up to 200MB)":
        uploaded_file = st.file_uploader(
            "Upload your video file",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Supported formats: MP4, MOV, AVI, MKV, WebM (max 200MB)"
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"Uploaded: **{uploaded_file.name}** ({file_size_mb:.1f} MB)")

            if st.button("Analyze Video", type="primary", use_container_width=True):
                temp_dir = tempfile.mkdtemp(prefix="video_analyzer_")
                try:
                    video_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    results = run_analysis(
                        video_path=video_path,
                        video_name=uploaded_file.name,
                        temp_dir=temp_dir,
                        analyze_visuals=analyze_visuals,
                        num_frames=num_frames,
                        vision_model=vision_model,
                        vision_model_name=vision_model_name,
                        analysis_model=analysis_model,
                        analysis_model_name=analysis_model_name,
                        custom_instructions=custom_instructions
                    )

                    # Store results in session state
                    st.session_state.analysis_results = results

                    # Rerun to display results properly
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                finally:
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

    else:
        st.markdown("""
        **Supported URL sources:**
        - Direct video URLs (.mp4, .mov, etc.)
        - Google Drive (share link)
        - Dropbox (share link)
        """)

        video_url = st.text_input(
            "Enter video URL",
            placeholder="https://drive.google.com/file/d/... or direct video URL"
        )

        if video_url:
            if st.button("Download & Analyze Video", type="primary", use_container_width=True):
                temp_dir = tempfile.mkdtemp(prefix="video_analyzer_")
                try:
                    download_status = st.empty()
                    download_progress = st.progress(0)

                    download_status.text("Downloading video...")

                    def update_download_progress(downloaded, total):
                        if total > 0:
                            pct = int((downloaded / total) * 100)
                            download_progress.progress(pct)
                            download_status.text(f"Downloading: {downloaded // (1024*1024)} MB / {total // (1024*1024)} MB")

                    video_path = download_video_from_url(
                        video_url,
                        temp_dir,
                        progress_callback=update_download_progress
                    )

                    video_name = os.path.basename(video_path)
                    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

                    download_status.empty()
                    download_progress.empty()
                    st.success(f"Downloaded: **{video_name}** ({file_size_mb:.1f} MB)")

                    results = run_analysis(
                        video_path=video_path,
                        video_name=video_name,
                        temp_dir=temp_dir,
                        analyze_visuals=analyze_visuals,
                        num_frames=num_frames,
                        vision_model=vision_model,
                        vision_model_name=vision_model_name,
                        analysis_model=analysis_model,
                        analysis_model_name=analysis_model_name,
                        custom_instructions=custom_instructions
                    )

                    # Store results in session state
                    st.session_state.analysis_results = results

                    # Rerun to display results properly
                    st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to download video: {str(e)}")
                    st.info("Make sure the URL is publicly accessible or a valid sharing link.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                finally:
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

    # Show placeholder when no input
    if st.session_state.analysis_results is None:
        st.markdown("---")
        st.markdown("### How it works")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**1. Upload/URL**")
            st.markdown("Provide your video file or URL")

        with col2:
            st.markdown("**2. Transcribe**")
            st.markdown("Audio transcribed via Whisper API")

        with col3:
            st.markdown("**3. Analyze**")
            st.markdown("AI analyzes content & visuals")

        with col4:
            st.markdown("**4. Results**")
            st.markdown("Get summary & actionable insights")


if __name__ == "__main__":
    main()
