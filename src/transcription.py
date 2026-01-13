"""
Transcription Service Module
Handles audio transcription using OpenAI Whisper API.
"""

import os
from pathlib import Path

from openai import OpenAI


def get_openai_client(api_key: str = None) -> OpenAI:
    """
    Get OpenAI client.

    Args:
        api_key: OpenAI API key (if None, will try st.secrets or env var)

    Returns:
        OpenAI client instance
    """
    if api_key is None:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OpenAI API key not found. Set it in Streamlit secrets or OPENAI_API_KEY env var.")

    return OpenAI(api_key=api_key)


def transcribe_audio(
    audio_path: str,
    api_key: str = None,
    language: str = None,
    include_timestamps: bool = True
) -> dict:
    """
    Transcribe audio file using OpenAI Whisper API.

    Args:
        audio_path: Path to the audio file
        api_key: OpenAI API key
        language: Language code (e.g., 'en', 'es') - auto-detect if None
        include_timestamps: Whether to include word-level timestamps

    Returns:
        Dictionary with transcript text and metadata
    """
    client = get_openai_client(api_key)

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    # Whisper API has 25MB limit - need to chunk larger files
    if file_size_mb > 24:
        return transcribe_large_audio(audio_path, api_key, language)

    with open(audio_path, "rb") as audio_file:
        # Use verbose_json for timestamps
        response_format = "verbose_json" if include_timestamps else "json"

        kwargs = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": response_format,
        }

        if language:
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

    result = {
        "text": response.text,
        "language": getattr(response, "language", language or "auto"),
        "duration": getattr(response, "duration", None),
    }

    # Include segments if available (for verbose_json)
    if hasattr(response, "segments"):
        result["segments"] = [
            {
                "start": seg.get("start", seg.start) if hasattr(seg, "start") else seg["start"],
                "end": seg.get("end", seg.end) if hasattr(seg, "end") else seg["end"],
                "text": seg.get("text", seg.text) if hasattr(seg, "text") else seg["text"],
            }
            for seg in response.segments
        ]

    return result


def transcribe_large_audio(
    audio_path: str,
    api_key: str = None,
    language: str = None,
    chunk_duration_minutes: int = 10
) -> dict:
    """
    Transcribe large audio files by splitting into chunks.

    Args:
        audio_path: Path to the audio file
        api_key: OpenAI API key
        language: Language code
        chunk_duration_minutes: Duration of each chunk in minutes

    Returns:
        Dictionary with combined transcript
    """
    import subprocess
    import tempfile

    client = get_openai_client(api_key)

    # Get audio duration
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())

    chunk_duration = chunk_duration_minutes * 60
    num_chunks = int(total_duration / chunk_duration) + 1

    all_text = []
    all_segments = []
    current_offset = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp3")

            # Extract chunk
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ss", str(start_time),
                "-t", str(chunk_duration),
                "-acodec", "libmp3lame",
                "-y",
                chunk_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Check if chunk exists and has content
            if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) < 1000:
                continue

            # Transcribe chunk
            with open(chunk_path, "rb") as audio_file:
                kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "verbose_json",
                }
                if language:
                    kwargs["language"] = language

                response = client.audio.transcriptions.create(**kwargs)

            all_text.append(response.text)

            # Adjust segment timestamps
            if hasattr(response, "segments"):
                for seg in response.segments:
                    adjusted_seg = {
                        "start": seg["start"] + start_time,
                        "end": seg["end"] + start_time,
                        "text": seg["text"],
                    }
                    all_segments.append(adjusted_seg)

    return {
        "text": " ".join(all_text),
        "language": language or "auto",
        "duration": total_duration,
        "segments": all_segments,
    }


def format_transcript_with_timestamps(segments: list) -> str:
    """
    Format transcript with timestamps for readability.

    Args:
        segments: List of segment dictionaries with start, end, text

    Returns:
        Formatted transcript string
    """
    lines = []

    for seg in segments:
        start = seg["start"]
        minutes = int(start // 60)
        seconds = int(start % 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        lines.append(f"{timestamp} {seg['text'].strip()}")

    return "\n".join(lines)
