"""
Video Processor Module
Handles audio extraction and key frame extraction from video files.
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_audio(video_path: str, output_dir: str = None) -> str:
    """
    Extract audio from video file as MP3.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the audio file (uses temp dir if None)

    Returns:
        Path to the extracted audio file
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    video_name = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{video_name}_audio.mp3")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "libmp3lame",
        "-ab", "128k",  # Bitrate
        "-ar", "16000",  # Sample rate (16kHz is good for speech)
        "-y",  # Overwrite output
        audio_path
    ]

    subprocess.run(cmd, capture_output=True, check=True)

    return audio_path


def extract_key_frames(
    video_path: str,
    output_dir: str = None,
    num_frames: int = 10,
    min_interval_seconds: float = 30.0
) -> List[Tuple[str, float]]:
    """
    Extract key frames from video at regular intervals.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save frames (uses temp dir if None)
        num_frames: Number of frames to extract
        min_interval_seconds: Minimum seconds between frames

    Returns:
        List of tuples (frame_path, timestamp_seconds)
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    video_name = Path(video_path).stem

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Calculate frame positions
    # Ensure minimum interval between frames
    interval = max(duration / (num_frames + 1), min_interval_seconds)
    actual_num_frames = min(num_frames, int(duration / min_interval_seconds))

    if actual_num_frames < 1:
        actual_num_frames = 1
        interval = duration / 2  # Get middle frame for short videos

    frames = []

    for i in range(actual_num_frames):
        # Calculate timestamp (skip first and last few seconds)
        timestamp = interval * (i + 1)

        if timestamp >= duration:
            break

        # Seek to frame
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if ret:
            # Save frame
            frame_path = os.path.join(
                output_dir,
                f"{video_name}_frame_{i:03d}_{int(timestamp)}s.jpg"
            )
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append((frame_path, timestamp))

    cap.release()

    return frames


def process_video(
    video_path: str,
    output_dir: str = None,
    extract_frames: bool = True,
    num_frames: int = 10
) -> dict:
    """
    Process video file: extract audio and optionally key frames.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save outputs
        extract_frames: Whether to extract key frames
        num_frames: Number of frames to extract

    Returns:
        Dictionary with audio_path, frames list, and video metadata
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_analyzer_")

    os.makedirs(output_dir, exist_ok=True)

    # Get video info
    duration = get_video_duration(video_path)

    result = {
        "video_path": video_path,
        "output_dir": output_dir,
        "duration_seconds": duration,
        "duration_formatted": format_duration(duration),
        "audio_path": None,
        "frames": []
    }

    # Extract audio
    result["audio_path"] = extract_audio(video_path, output_dir)

    # Extract frames if requested
    if extract_frames:
        result["frames"] = extract_key_frames(
            video_path,
            output_dir,
            num_frames=num_frames
        )

    return result


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
