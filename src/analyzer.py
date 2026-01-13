"""
Analysis Engine Module
Generates executive summaries and actionable insights from video content.
"""

import os
from typing import List, Optional

from anthropic import Anthropic


def get_anthropic_client(api_key: str = None) -> Anthropic:
    """
    Get Anthropic client.

    Args:
        api_key: Anthropic API key (if None, will try st.secrets or env var)

    Returns:
        Anthropic client instance
    """
    if api_key is None:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("Anthropic API key not found. Set it in Streamlit secrets or ANTHROPIC_API_KEY env var.")

    return Anthropic(api_key=api_key)


def generate_analysis(
    transcript: str,
    visual_summary: Optional[str] = None,
    video_duration: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    api_key: str = None
) -> dict:
    """
    Generate comprehensive video analysis including executive summary and actionable insights.

    Args:
        transcript: Full transcript of the video
        visual_summary: Optional summary of visual content
        video_duration: Optional formatted duration string
        custom_instructions: Optional additional instructions for analysis
        api_key: Anthropic API key

    Returns:
        Dictionary containing all analysis components
    """
    client = get_anthropic_client(api_key)

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

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    analysis_text = response.content[0].text

    # Parse the analysis into sections
    sections = parse_analysis_sections(analysis_text)

    return {
        "full_analysis": analysis_text,
        "sections": sections,
        "model": "claude-sonnet-4-20250514",
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


def generate_quick_summary(transcript: str, api_key: str = None) -> str:
    """
    Generate a quick one-paragraph summary of the video.

    Args:
        transcript: Full transcript of the video
        api_key: Anthropic API key

    Returns:
        One-paragraph summary string
    """
    client = get_anthropic_client(api_key)

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

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
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

    return response.content[0].text


def generate_topic_tags(transcript: str, api_key: str = None, max_tags: int = 10) -> List[str]:
    """
    Generate topic tags for the video content.

    Args:
        transcript: Full transcript of the video
        api_key: Anthropic API key
        max_tags: Maximum number of tags to generate

    Returns:
        List of topic tags
    """
    client = get_anthropic_client(api_key)

    # Use a sample for long transcripts
    sample = transcript[:20000] if len(transcript) > 20000 else transcript

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
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
        for tag in response.content[0].text.strip().split("\n")
        if tag.strip()
    ]

    return tags[:max_tags]
