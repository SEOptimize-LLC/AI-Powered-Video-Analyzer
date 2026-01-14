"""
Visual Analyzer Module
Analyzes video frames using OpenRouter API with vision-capable models.
"""

import base64
import os
from typing import List, Tuple

from openai import OpenAI


# Available vision-capable models on OpenRouter
VISION_MODELS = {
    "gemini-flash": "google/gemini-3-flash-preview",
    "claude-sonnet": "anthropic/claude-sonnet-4.5",
    "gpt-5-mini": "openai/gpt-5-mini",
}

DEFAULT_VISION_MODEL = "google/gemini-3-flash-preview"


def get_openrouter_client(api_key: str = None) -> OpenAI:
    """
    Get OpenRouter client (OpenAI-compatible).

    Args:
        api_key: OpenRouter API key (if None, will try st.secrets or env var)

    Returns:
        OpenAI client configured for OpenRouter
    """
    if api_key is None:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OpenRouter API key not found. Set it in Streamlit secrets or OPENROUTER_API_KEY env var.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def encode_image_to_base64(image_path: str) -> str:
    """Read image file and encode to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get media type from image file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def analyze_single_frame(
    image_path: str,
    timestamp: float,
    api_key: str = None,
    model: str = None,
    context: str = None
) -> dict:
    """
    Analyze a single video frame.

    Args:
        image_path: Path to the frame image
        timestamp: Timestamp in seconds where frame was captured
        api_key: OpenRouter API key
        model: Model to use (defaults to Gemini Flash)
        context: Optional context about the video

    Returns:
        Dictionary with frame analysis
    """
    client = get_openrouter_client(api_key)
    model = model or DEFAULT_VISION_MODEL

    image_data = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    timestamp_str = f"{minutes:02d}:{seconds:02d}"

    prompt = f"""Analyze this video frame captured at timestamp {timestamp_str}.

Describe:
1. What is being shown (slides, diagrams, people, screen recordings, etc.)
2. Any visible text, titles, or key information
3. Visual elements that add context to the content

Keep your analysis concise but informative (2-4 sentences)."""

    if context:
        prompt = f"Context: {context}\n\n{prompt}"

    response = client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    return {
        "timestamp": timestamp,
        "timestamp_formatted": timestamp_str,
        "image_path": image_path,
        "analysis": response.choices[0].message.content,
        "model": model,
    }


def analyze_frames(
    frames: List[Tuple[str, float]],
    api_key: str = None,
    model: str = None,
    context: str = None,
    progress_callback=None
) -> List[dict]:
    """
    Analyze multiple video frames.

    Args:
        frames: List of (image_path, timestamp) tuples
        api_key: OpenRouter API key
        model: Model to use for analysis
        context: Optional context about the video
        progress_callback: Optional callback function(current, total) for progress

    Returns:
        List of frame analysis dictionaries
    """
    results = []

    for i, (image_path, timestamp) in enumerate(frames):
        if progress_callback:
            progress_callback(i + 1, len(frames))

        analysis = analyze_single_frame(
            image_path,
            timestamp,
            api_key=api_key,
            model=model,
            context=context
        )
        results.append(analysis)

    return results


def summarize_visual_content(
    frame_analyses: List[dict],
    api_key: str = None,
    model: str = None
) -> str:
    """
    Create a summary of all visual content from frame analyses.

    Args:
        frame_analyses: List of frame analysis dictionaries
        api_key: OpenRouter API key
        model: Model to use for summarization

    Returns:
        Summary string of visual content
    """
    if not frame_analyses:
        return "No visual content analyzed."

    client = get_openrouter_client(api_key)
    model = model or DEFAULT_VISION_MODEL

    # Compile all frame analyses
    content_parts = []
    for analysis in frame_analyses:
        content_parts.append(
            f"[{analysis['timestamp_formatted']}] {analysis['analysis']}"
        )

    visual_content = "\n\n".join(content_parts)

    response = client.chat.completions.create(
        model=model,
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"""Based on these video frame analyses, provide a brief summary of the visual content and presentation style of the video:

{visual_content}

Summarize in 2-3 paragraphs:
1. The visual format/style of the video (presentation slides, talking head, screen recording, etc.)
2. Key visual elements or information shown
3. How the visuals support the content""",
            }
        ],
    )

    return response.choices[0].message.content
