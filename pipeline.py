"""
YouTube Shorts Generation Pipeline
----------------------------------
Automates generating viral YouTube Shorts:
1. Script & metadata generation using Gemini (google-genai) with Pydantic structured output.
   - High-retention 35-to-45-second script with a scroll-stopping hook in the first 3 seconds.
   - Punchy title and exactly 5 hashtags.
2. Text-to-Speech audio generation using edge-tts (voice: en-US-ChristopherNeural).
3. Rotating background video library:
   - Scans 'backgrounds/' and randomly selects a gameplay/background video (e.g., Minecraft, GTA, Subway Surfers).
   - Uses ffprobe to detect total background video duration.
   - Calculates a random seek offset (-ss) with sufficient remaining duration to cover voice.mp3.
   - Scales and center-crops to strictly vertical 1080x1920.
   - Mutes all original game audio so only voice.mp3 is audible.
   - Cuts video at the exact end of voice.mp3.
4. Outputs:
   - Saves final video to 'outputs/short_YYYYMMDD_HHMMSS.mp4'.
   - Saves metadata to 'outputs/short_YYYYMMDD_HHMMSS.json'.
"""

import os
import re
import sys
import json
import random
import shutil
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import edge_tts


class ShortContent(BaseModel):
    title: str = Field(description="A viral, intriguing YouTube Short title (max 60 characters)")
    tags: list[str] = Field(
        description="Exactly 5 high-traffic hashtags (e.g. ['#mystery', '#facts', '#science', '#history', '#shorts'])"
    )
    script: str = Field(
        description="A 35-to-45-second punchy, high-retention spoken narration with a magnetic hook in the first 3 seconds. Spoken words only."
    )


def resolve_ffmpeg_binaries() -> tuple[str, str]:
    """Finds ffmpeg and ffprobe binaries in PATH, registry, or WinGet package directories."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            user_path, _ = winreg.QueryValueEx(key, "Path")
            for p in user_path.split(";"):
                if p and p not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = os.environ.get("PATH", "") + ";" + p
    except Exception:
        pass

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_dir.exists():
            for p in winget_dir.glob("**/ffmpeg.exe"):
                bin_dir = str(p.parent)
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + ";" + os.environ.get("PATH", "")
                break

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    return ffmpeg_bin, ffprobe_bin


def get_media_duration(file_path: str, ffprobe_bin: str = "ffprobe") -> float | None:
    """Gets exact media duration in seconds using ffprobe."""
    try:
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return None


def get_background_videos(backgrounds_dir: str = "backgrounds") -> list[Path]:
    """
    Scans the 'backgrounds/' directory for video files (.mp4, .webm, .mkv, .mov).
    Raises a clear error if none are found.
    """
    bg_folder = Path(backgrounds_dir)
    if not bg_folder.exists():
        bg_folder.mkdir(parents=True, exist_ok=True)

    supported_exts = {".mp4", ".webm", ".mkv", ".mov"}
    videos = [p for p in bg_folder.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]

    if not videos:
        raise FileNotFoundError(
            f"\n[ERROR] No background video files found in '{backgrounds_dir}/'!\n"
            f"Please drop your gameplay or background clips (e.g. minecraft.mp4, gta.mp4, subway.mp4) "
            f"into '{bg_folder.resolve()}'."
        )
    return videos


def select_random_background(backgrounds_dir: str = "backgrounds") -> Path:
    """Randomly chooses one video from the backgrounds/ directory."""
    videos = get_background_videos(backgrounds_dir)
    chosen = random.choice(videos)
    print(f"[1/5] Background Selected: '{chosen.name}' (from {len(videos)} available in '{backgrounds_dir}/')")
    return chosen


def calculate_seek_offset(bg_duration: float | None, voice_duration: float) -> float:
    """
    Calculates a random starting point (seek offset) so each generated video uses
    a different section of the gameplay, ensuring enough time remains to cover voice.mp3.
    """
    if bg_duration is None or bg_duration <= voice_duration:
        print("[Seek Offset] Background duration <= narration duration; starting from 0.0s (looping enabled).")
        return 0.0

    max_start = bg_duration - voice_duration
    offset = round(random.uniform(0.0, max_start), 2)
    print(
        f"[Seek Offset] Video duration: {bg_duration:.2f}s | Audio duration: {voice_duration:.2f}s\n"
        f"              Random seek offset: {offset:.2f}s (valid range: 0.00s to {max_start:.2f}s)"
    )
    return offset


def generate_short_content(api_key: str, model: str = "gemini-2.5-flash") -> ShortContent:
    """
    Calls Gemini using google-genai SDK with Pydantic structured output.
    Prompt enforces a punchy 35-to-45-second script with a 3-second scroll-stopping hook,
    title, and exactly 5 hashtags.
    """
    from google import genai
    from google.genai import types

    print(f"[2/5] Generating viral script with Gemini ({model})...")
    client = genai.Client(api_key=api_key)

    prompt = (
        "You are an elite viral YouTube Shorts scriptwriter specializing in high-retention storytelling.\n\n"
        "Write a punchy, mind-bending 35-to-45-second narration about an astonishing scientific fact, historical mystery, or cosmic wonder.\n\n"
        "STRUCTURE REQUIREMENTS:\n"
        "1. HOOK (0 to 3 seconds): Start immediately with a shocking statement, impossible question, or suspenseful contradiction that prevents the viewer from scrolling away. Do NOT say 'Did you know' or 'Hey guys'.\n"
        "2. THE BUILD & ESCALATION (3 to 35 seconds): Rapid-fire, vivid explanation that raises the stakes with each sentence.\n"
        "3. PUNCHLINE / CLIMAX (35 to 45 seconds): A jaw-dropping conclusion that lingers in the viewer's mind and invites them to comment.\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Script must be spoken words ONLY. Absolutely NO sound effects, audio brackets (e.g. [suspenseful music]), markdown asterisks, or speaker labels.\n"
        "- Word count: strictly between 90 and 115 words so it takes between 35 and 45 seconds at natural spoken narration pace.\n"
        "- Provide exactly 5 relevant hashtags."
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ShortContent,
        ),
    )

    if hasattr(response, "parsed") and response.parsed:
        if isinstance(response.parsed, ShortContent):
            return response.parsed
        return ShortContent.model_validate(response.parsed)

    return ShortContent.model_validate_json(response.text)


async def generate_voiceover(
    script: str,
    output_path: str = "voice.mp3",
    voice: str = "en-US-ChristopherNeural"
) -> str:
    """Converts script to voice.mp3 using edge-tts."""
    print(f"[3/5] Synthesizing speech with edge-tts ({voice})...")
    communicate = edge_tts.Communicate(text=script, voice=voice)
    await communicate.save(output_path)
    print(f"[3/5] Audio saved: {output_path}")
    return output_path


def render_short_video(
    bg_video_path: str,
    voice_audio_path: str,
    output_video_path: str,
    seek_offset: float = 0.0,
    voice_duration: float | None = None,
    metadata: dict = None,
    metadata_output_path: str = None,
    ffmpeg_bin: str = "ffmpeg",
):
    """
    FFmpeg Rendering:
    - Seeks to random timestamp (-ss).
    - Ensures output is strictly vertical 1080x1920 (crop/scale).
    - Mutes original game audio, mapping only voice.mp3.
    - Cuts video at the exact end of voice.mp3 (-shortest / -t).
    """
    print(f"[4/5] Rendering 1080x1920 vertical video with FFmpeg...")

    # Strict 9:16 vertical crop filter (scale up to fill 1080x1920, center-crop, force 1:1 SAR)
    vf_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1"

    cmd = [
        ffmpeg_bin,
        "-y",
    ]

    # Fast seek to random gameplay offset
    if seek_offset > 0:
        cmd.extend(["-ss", str(seek_offset)])

    # Input background video looped
    cmd.extend([
        "-stream_loop", "-1",
        "-i", bg_video_path,
        "-i", voice_audio_path,
        # Strip all game audio: only map video from input 0 and audio from input 1 (voice.mp3)
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
    ])

    # Cut video at exact end of voiceover
    if voice_duration is not None:
        cmd.extend(["-t", str(voice_duration)])
    else:
        cmd.append("-shortest")

    cmd.extend(["-pix_fmt", "yuv420p", output_video_path])

    subprocess.run(cmd, check=True)
    print(f"[4/5] Video rendered: {output_video_path}")

    if metadata is not None and metadata_output_path is not None:
        with open(metadata_output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[5/5] Metadata saved: {metadata_output_path}")


def run_pipeline():
    load_dotenv()
    ffmpeg_bin, ffprobe_bin = resolve_ffmpeg_binaries()
    print(f"FFmpeg binary : {ffmpeg_bin}")
    print(f"FFprobe binary: {ffprobe_bin}\n")

    # 1. Random background selection
    chosen_bg = select_random_background("backgrounds")

    # 2. Total duration of background video
    bg_duration = get_media_duration(str(chosen_bg), ffprobe_bin=ffprobe_bin)
    if bg_duration:
        print(f"    Background Total Duration: {bg_duration:.2f} seconds\n")
    else:
        print("    Background Duration: Unknown (defaulting seek to 0.0s)\n")

    # 3. Gemini script generation (or mock mode)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    mock_mode = "--mock" in sys.argv or os.getenv("PIPELINE_MOCK", "").lower() in ("1", "true")

    if mock_mode or not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
        if not mock_mode:
            print("[INFO] GEMINI_API_KEY not configured in .env. Using high-retention template script.")
            print("       (Add your GEMINI_API_KEY to .env to generate unique AI scripts on every run.)\n")
        else:
            print("[INFO] Mock mode activated.\n")

        content = ShortContent(
            title="The Dark Secret of the Mariana Trench",
            tags=["#ocean", "#mystery", "#science", "#deepsea", "#shorts"],
            script=(
                "Seven miles beneath the ocean surface, the pressure is strong enough to crush a titanium submarine like a soda can. "
                "Yet when cameras finally descended into the Challenger Deep, they didn't find a barren wasteland. "
                "They discovered ghostly, translucent creatures thriving in total darkness, surviving on chemical vents hotter than molten lead. "
                "Even more chilling, scientists detected strange rhythmic metallic sounds reverberating through the abyss that don't match any known geological activity. "
                "We know more about the surface of Mars than the bottom of our own oceans. "
                "What do you think is watching us from down there?"
            ),
        )
    else:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        content = generate_short_content(api_key=gemini_api_key, model=model_name)

    print("--- Generated YouTube Short ---")
    print(f"Title   : {content.title}")
    print(f"Hashtags: {' '.join(content.tags)}")
    print(f"Script  : {content.script}\n")

    # 4. Text-to-Speech
    voice_path = "voice.mp3"
    voice_name = os.getenv("TTS_VOICE", "en-US-ChristopherNeural")
    asyncio.run(generate_voiceover(content.script, output_path=voice_path, voice=voice_name))

    # 5. Measure narration duration & compute seek offset
    voice_duration = get_media_duration(voice_path, ffprobe_bin=ffprobe_bin)
    if voice_duration:
        print(f"Narration Duration: {voice_duration:.2f} seconds")

    seek_offset = calculate_seek_offset(bg_duration, voice_duration or 40.0)

    # 6. Setup output paths in 'outputs/' (short_YYYYMMDD_HHMMSS.mp4)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = str(outputs_dir / f"short_{timestamp}.mp4")
    output_metadata_path = str(outputs_dir / f"short_{timestamp}.json")

    metadata = {
        "title": content.title,
        "tags": content.tags,
        "script": content.script,
        "voice": voice_name,
        "background_clip": chosen_bg.name,
        "background_total_duration_sec": bg_duration,
        "seek_offset_seconds": seek_offset,
        "video_duration_seconds": round(voice_duration, 2) if voice_duration else None,
        "video_resolution": "1080x1920 (9:16)",
        "output_file": output_video_path,
        "timestamp": timestamp,
    }

    # 7. Render final short
    render_short_video(
        bg_video_path=str(chosen_bg),
        voice_audio_path=voice_path,
        output_video_path=output_video_path,
        seek_offset=seek_offset,
        voice_duration=voice_duration,
        metadata=metadata,
        metadata_output_path=output_metadata_path,
        ffmpeg_bin=ffmpeg_bin,
    )

    print("\n Execution completed successfully!")
    print(f"  Final Short   : {output_video_path}")
    print(f"  Metadata File : {output_metadata_path}")


if __name__ == "__main__":
    run_pipeline()
