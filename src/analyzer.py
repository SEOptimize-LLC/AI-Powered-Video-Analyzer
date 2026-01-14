"""
Analysis Engine Module
Generates executive summaries and actionable insights using OpenRouter API.
"""

import os
from typing import List, Optional

from openai import OpenAI


# Available models on OpenRouter for analysis
ANALYSIS_MODELS = {
    "claude-sonnet": "anthropic/claude-sonnet-4.5",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gemini-flash": "google/gemini-3-flash-preview",
}

DEFAULT_ANALYSIS_MODEL = "anthropic/claude-sonnet-4.5"


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


def generate_analysis(
    transcript: str,
    visual_summary: Optional[str] = None,
    video_duration: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    api_key: str = None,
    model: str = None
) -> dict:
    """
    Generate comprehensive video analysis including executive summary and actionable insights.

    Args:
        transcript: Full transcript of the video
        visual_summary: Optional summary of visual content
        video_duration: Optional formatted duration string
        custom_instructions: Optional additional instructions for analysis
        api_key: OpenRouter API key
        model: Model to use for analysis

    Returns:
        Dictionary containing all analysis components
    """
    client = get_openrouter_client(api_key)
    model = model or DEFAULT_ANALYSIS_MODEL

    # Build context
    context_parts = []

    if video_duration:
        context_parts.append(f"Video Duration: {video_duration}")

    if visual_summary:
        context_parts.append(f"Visual Content Summary:\n{visual_summary}")

    context = "\n\n".join(context_parts) if context_parts else ""

    # Build the analysis prompt
    system_prompt = """You are an expert content analyst specializing in extracting valuable insights from video content. Your role is to help busy professionals save time by providing comprehensive, actionable summaries of video content.

Your analysis should be:
- Thorough yet concise
- Focused on practical value
- Well-structured and easy to scan
- Actionable and specific"""

    user_prompt = f"""Analyze the following video transcript and provide a comprehensive analysis.

{f"CONTEXT:{chr(10)}{context}{chr(10)}{chr(10)}" if context else ""}TRANSCRIPT:
{transcript}

{f"ADDITIONAL INSTRUCTIONS:{chr(10)}{custom_instructions}{chr(10)}{chr(10)}" if custom_instructions else ""}

Please provide your analysis in the following format:

## Executive Summary
A concise 2-3 paragraph summary that captures the essence of the video content. Include the main topic, key arguments, and overall conclusions.

## Key Topics Covered
List the main topics/themes discussed in the video with brief descriptions.

## Key Insights
The most important insights, findings, or lessons from the video. Be specific and include context.

## Actionable Takeaways
Specific, practical actions the viewer can take based on the content. Make these concrete and implementable.

## Notable Quotes
Any particularly impactful or quotable statements from the video (with approximate timestamps if available in the transcript).

## Questions Raised
Important questions raised in the video or questions the viewer might want to explore further.

## Related Topics for Further Research
Suggest related topics, concepts, or resources the viewer might want to explore to deepen their understanding.

## Content Overview
A brief structural overview of how the video is organized (e.g., "Introduction (0-5 min), Main Concepts (5-30 min), Case Studies (30-45 min), Conclusion (45-60 min)")"""

    response = client.chat.completions.create(
        model=model,
        max_tokens=4000,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    analysis_text = response.choices[0].message.content

    # Parse the analysis into sections
    sections = parse_analysis_sections(analysis_text)

    return {
        "full_analysis": analysis_text,
        "sections": sections,
        "model": model,
        "transcript_length": len(transcript),
    }


def parse_analysis_sections(analysis_text: str) -> dict:
    """
    Parse the analysis text into individual sections.

    Args:
        analysis_text: Full analysis text with markdown headers

    Returns:
        Dictionary with section names as keys and content as values
    """
    sections = {}
    current_section = None
    current_content = []

    for line in analysis_text.split("\n"):
        if line.startswith("## "):
            # Save previous section
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()

            # Start new section
            current_section = line[3:].strip()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def generate_quick_summary(
    transcript: str,
    api_key: str = None,
    model: str = None
) -> str:
    """
    Generate a quick one-paragraph summary of the video.

    Args:
        transcript: Full transcript of the video
        api_key: OpenRouter API key
        model: Model to use

    Returns:
        One-paragraph summary string
    """
    client = get_openrouter_client(api_key)
    model = model or DEFAULT_ANALYSIS_MODEL

    # For very long transcripts, use a sample
    if len(transcript) > 50000:
        # Take beginning, middle, and end
        part_size = 15000
        beginning = transcript[:part_size]
        middle_start = len(transcript) // 2 - part_size // 2
        middle = transcript[middle_start:middle_start + part_size]
        end = transcript[-part_size:]
        transcript_sample = f"{beginning}\n\n[...]\n\n{middle}\n\n[...]\n\n{end}"
    else:
        transcript_sample = transcript

    response = client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""Provide a concise one-paragraph summary (3-5 sentences) of this video transcript. Focus on the main topic and key takeaway.

TRANSCRIPT:
{transcript_sample}""",
            }
        ],
    )

    return response.choices[0].message.content


def generate_topic_tags(
    transcript: str,
    api_key: str = None,
    model: str = None,
    max_tags: int = 10
) -> List[str]:
    """
    Generate topic tags for the video content.

    Args:
        transcript: Full transcript of the video
        api_key: OpenRouter API key
        model: Model to use
        max_tags: Maximum number of tags to generate

    Returns:
        List of topic tags
    """
    client = get_openrouter_client(api_key)
    model = model or DEFAULT_ANALYSIS_MODEL

    # Use a sample for long transcripts
    sample = transcript[:20000] if len(transcript) > 20000 else transcript

    response = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"""Generate {max_tags} topic tags for this video transcript. Return only the tags, one per line, no numbers or bullets.

TRANSCRIPT:
{sample}""",
            }
        ],
    )

    tags = [
        tag.strip().strip("-•*")
        for tag in response.choices[0].message.content.strip().split("\n")
        if tag.strip()
    ]

    return tags[:max_tags]
