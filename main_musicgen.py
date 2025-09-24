from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List
import numpy as np
import random
import io
import wave
import os
import time
import requests

# ----------------------------------------------------------------------------
# TuneCrafter backend with MusicGen integration
#
# This module exposes a simple API for generating lyrics and audio tracks.
#
# Endpoints:
# * POST /api/generateLyrics – returns four lines of deterministic placeholder
#   lyrics based on the provided prompt and genre. Replace the
#   ``create_lyrics`` function with a call to a language model (e.g. GPT‑4,
#   Llama) for higher‑quality lyrics.
#
# * POST /api/generateSong – attempts to create a song using a MusicGen
#   service. The MusicGen service URL and optional API key must be provided
#   via the ``MUSICGEN_API_URL`` and ``MUSICGEN_API_KEY`` environment
#   variables. If the call fails, the function falls back to the Stable
#   Audio API (when the ``STABLE_AUDIO_API_KEY`` is present) and finally
#   falls back to a procedurally generated melody.
#
# * GET /api/health – returns a simple health status.

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/generateLyrics")
async def generate_lyrics(payload: Dict[str, str]):
    """Generate simple lyrics from a prompt and genre.

    Parameters
    ----------
    payload: Dict[str, str]
        A JSON body containing ``prompt`` and optional ``genre`` keys.

    Returns
    -------
    JSONResponse
        A JSON object with a ``lyrics`` field.
    """
    prompt = payload.get("prompt", "")
    genre = payload.get("genre", "Unknown")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    lyrics = create_lyrics(prompt, genre)
    return JSONResponse(content={"lyrics": lyrics})


def create_lyrics(prompt: str, genre: str) -> str:
    """Create deterministic placeholder lyrics.

    This helper splits the prompt into words, falling back to the genre name
    when no valid words are present. It assembles four lines of lyrics by
    mixing prompt words with genre‑specific filler words. The use of a
    deterministic seed ensures repeatable output for the same prompt and genre.

    Parameters
    ----------
    prompt: str
        User‑provided description.
    genre: str
        Selected genre.

    Returns
    -------
    str
        A four‑line string representing simple lyrics.
    """
    words = [w.strip(".,!?;:") for w in prompt.lower().split() if w.strip(".,!?;:")]
    if not words:
        words = [genre.lower()]
    seed = hash((prompt, genre))
    rng = random.Random(seed)
    filler_by_genre = {
        "pop": ["love", "dance", "night", "heart", "forever"],
        "rock": ["fire", "wild", "road", "burn", "freedom"],
        "hip-hop": ["flow", "beat", "street", "rise", "dream"],
        "country": ["road", "home", "whiskey", "truck", "sunset"],
        "electronic": ["synth", "beat", "pulse", "vibe", "future"],
        "jazz": ["swing", "soul", "blue", "smooth", "groove"],
        "r&b": ["heart", "groove", "soul", "touch", "midnight"],
        "latin": ["amor", "ritmo", "baila", "corazón", "pasión"],
    }
    fillers = filler_by_genre.get(genre.lower(), ["story", "melody", "rhythm", "beat", "feeling"])
    lines: List[str] = []
    for _ in range(4):
        line_words = []
        length = rng.randint(4, 8)
        for _ in range(length):
            if rng.random() < 0.3:
                line_words.append(rng.choice(fillers))
            else:
                line_words.append(rng.choice(words))
        if line_words:
            line_words[0] = line_words[0].capitalize()
        lines.append(" ".join(line_words))
    return "\n".join(lines)


def generate_melody(duration: float, genre: str, sample_rate: int = 44100) -> np.ndarray:
    """Generate a simple sine‑wave melody as a fallback.

    The melody cycles through a set of base frequencies for the given genre,
    switching notes every half second.
    """
    freqs_by_genre = {
        "pop": [261.63, 329.63, 392.00, 523.25],
        "rock": [82.41, 110.00, 146.83, 196.00],
        "hip-hop": [98.00, 123.47, 164.81, 196.00],
        "country": [130.81, 146.83, 196.00, 261.63],
        "electronic": [440.00, 554.37, 659.25, 880.00],
        "jazz": [196.00, 233.08, 293.66, 349.23],
        "r&b": [174.61, 220.00, 261.63, 349.23],
        "latin": [196.00, 246.94, 329.63, 392.00],
    }
    base_freqs = freqs_by_genre.get(genre.lower(), [261.63, 329.63, 392.00, 523.25])
    num_samples = int(duration * sample_rate)
    note_length = int(0.5 * sample_rate)
    samples = np.zeros(num_samples, dtype=np.float32)
    for start in range(0, num_samples, note_length):
        freq = random.choice(base_freqs)
        end = min(start + note_length, num_samples)
        t = np.arange(start, end)
        samples[start:end] = 0.5 * np.sin(2 * np.pi * freq * t / sample_rate)
    return samples


def to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 array into WAV bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_data = np.int16(np.clip(audio_array, -1, 1) * 32767).tobytes()
        wav_file.writeframes(wav_data)
    buffer.seek(0)
    return buffer.read()


def generate_musicgen_audio(prompt: str, duration: float) -> bytes | None:
    """Call a remote MusicGen API to generate audio.

    The MusicGen service must be exposed via the ``MUSICGEN_API_URL``
    environment variable. Optionally, an API key can be provided via
    ``MUSICGEN_API_KEY`` which will be passed in the ``Authorization``
    header. The JSON payload includes the prompt and duration. If the
    request succeeds (HTTP 200), the raw audio bytes are returned. On
    error, ``None`` is returned.
    """
    api_url = os.environ.get("MUSICGEN_API_URL")
    if not api_url:
        return None
    headers: Dict[str, str] = {}
    api_key = os.environ.get("MUSICGEN_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = {"prompt": prompt, "duration": duration}
    try:
        resp = requests.post(api_url, json=data, headers=headers, timeout=120)
        if resp.status_code == 200 and resp.content:
            return resp.content
        return None
    except Exception:
        return None


def generate_stable_audio(prompt: str, duration: float) -> bytes | None:
    """Call the Stable Audio API to generate audio.

    The API key is read from ``STABLE_AUDIO_API_KEY``. If the key is not
    provided or any error occurs, returns ``None``.
    """
    api_key = os.environ.get("STABLE_AUDIO_API_KEY")
    if not api_key:
        return None
    endpoint = "https://api.aimlapi.com/v2/generate/audio"
    headers = {"access-token": api_key}
    data = {
        "model": "stable-audio",
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration,
    }
    try:
        resp = requests.post(endpoint, json=data, headers=headers, timeout=30)
        resp.raise_for_status()
        gen_id = resp.json().get("id")
        if not gen_id:
            return None
        # Poll for up to 60 seconds
        for _ in range(20):
            time.sleep(3)
            status_resp = requests.get(f"{endpoint}?generation_id={gen_id}", headers=headers, timeout=30)
            if status_resp.status_code != 200:
                continue
            status_json = status_resp.json()
            if status_json.get("status") == "succeeded" and status_json.get("audio_file"):
                audio_url = status_json["audio_file"]["url"]
                audio_file_resp = requests.get(audio_url, timeout=60)
                if audio_file_resp.status_code == 200:
                    return audio_file_resp.content
        return None
    except Exception:
        return None


@app.post("/api/generateSong")
async def generate_song(payload: Dict[str, str]):
    """Generate a song using MusicGen, Stable Audio or a fallback melody.

    The request payload must include ``duration`` and may include
    ``lyrics``, ``prompt``, ``genre`` and ``vocal``. MusicGen is tried first
    (if configured), then Stable Audio, then a procedural melody.
    """
    genre = payload.get("genre", "Unknown")
    duration = payload.get("duration")
    lyrics = payload.get("lyrics")
    vocal = payload.get("vocal", "")
    if duration is None:
        raise HTTPException(status_code=400, detail="Duration is required")
    try:
        duration = float(duration)
    except ValueError:
        raise HTTPException(status_code=400, detail="Duration must be a number")

    prompt: str | None = None
    if lyrics:
        prompt = f"{lyrics} in {genre} style {vocal} vocals"
    elif payload.get("prompt"):
        prompt = f"{payload.get('prompt')} {genre} {vocal}"

    audio_bytes: bytes | None = None
    if prompt:
        # Try MusicGen first
        audio_bytes = generate_musicgen_audio(prompt, duration)
        if not audio_bytes:
            # Fall back to Stable Audio if available
            audio_bytes = generate_stable_audio(prompt, duration)

    if audio_bytes:
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tune_{genre.lower()}_{int(duration)}s.wav"
            },
        )

    # Fallback: generate a simple melody if neither API succeeded
    sample_rate = 44100
    audio_array = generate_melody(duration, genre, sample_rate)
    wav_bytes = to_wav_bytes(audio_array, sample_rate)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=tune_{genre.lower()}_{int(duration)}s.wav"
        },
    )


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}