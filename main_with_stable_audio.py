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
# TuneCrafter backend with Stable Audio integration
#
# This module provides two main endpoints:
#
#  * POST /api/generateLyrics  – generate simple placeholder lyrics from a text
#    prompt and genre.  This uses a deterministic algorithm with genre-specific
#    filler words and random mixing.  Replace the implementation of
#    `create_lyrics` with a call to a language model for better results.
#
#  * POST /api/generateSong – generate a song from provided lyrics and
#    parameters.  If a Stable Audio API key is configured via the
#    STABLE_AUDIO_API_KEY environment variable, this endpoint will attempt to
#    generate high-quality audio via the Stable Audio API.  Otherwise it
#    falls back to a procedural sine-wave melody.  See the README for
#    further instructions on configuring the API key.
#
#  * GET /api/health – simple health-check endpoint.

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
        A JSON body containing a ``prompt`` string and optional ``genre``.

    Returns
    -------
    JSONResponse
        A JSON object with a ``lyrics`` field containing the generated text.
    """
    prompt = payload.get("prompt", "")
    genre = payload.get("genre", "Unknown")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    lyrics = create_lyrics(prompt, genre)
    return JSONResponse(content={"lyrics": lyrics})


def create_lyrics(prompt: str, genre: str) -> str:
    """Create placeholder lyrics based on a prompt and genre.

    This function splits the prompt into words, falls back to the genre name
    when no words are provided, and then builds four lines of lyrics by
    mixing prompt words with genre-specific filler words.  The output is
    deterministic for a given prompt and genre.

    Parameters
    ----------
    prompt: str
        The user's description of their song.
    genre: str
        The selected genre.

    Returns
    -------
    str
        A four-line lyrics string separated by newlines.
    """
    words = [w.strip(".,!?;:") for w in prompt.lower().split() if w.strip(".,!?;:")]
    if not words:
        words = [genre.lower()]
    # Seed the RNG with a hash to produce repeatable results
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
        "latin": ["amor", "ritmo", "baila", "corazon", "pasion"],
    }
    fillers = filler_by_genre.get(
        genre.lower(), ["story", "melody", "rhythm", "beat", "feeling"]
    )
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
    """Generate a simple sine-wave melody as a fallback.

    The melody cycles through a set of base frequencies for the given
    genre, switching notes every half-second.

    Parameters
    ----------
    duration: float
        Length of the song in seconds.
    genre: str
        The selected genre.
    sample_rate: int, optional
        Audio sample rate.  Defaults to 44100 Hz.

    Returns
    -------
    np.ndarray
        A one-dimensional float32 array of audio samples in the range
        [-1.0, 1.0].
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
    base_freqs = freqs_by_genre.get(
        genre.lower(), [261.63, 329.63, 392.00, 523.25]
    )
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
    """Convert a float32 array into WAV byte content."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_data = np.int16(np.clip(audio_array, -1, 1) * 32767).tobytes()
        wav_file.writeframes(wav_data)
    buffer.seek(0)
    return buffer.read()


def generate_stable_audio(prompt: str, duration: float) -> bytes | None:
    """Call the Stable Audio API to generate a song.

    The API key must be provided via the ``STABLE_AUDIO_API_KEY`` environment
    variable.  If no key is available or if any request fails, this
    function returns ``None``.

    Parameters
    ----------
    prompt: str
        A combined string including the user's lyrics, genre and vocal style.
    duration: float
        Desired length of the output audio in seconds.

    Returns
    -------
    bytes or None
        Audio data as bytes if successful, otherwise ``None``.
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
        # Poll for completion up to ~60 seconds
        for _ in range(20):
            time.sleep(3)
            status_resp = requests.get(
                f"{endpoint}?generation_id={gen_id}",
                headers=headers,
                timeout=30,
            )
            if status_resp.status_code != 200:
                continue
            status_json = status_resp.json()
            if (
                status_json.get("status") == "succeeded"
                and status_json.get("audio_file")
            ):
                audio_url = status_json["audio_file"]["url"]
                audio_file_resp = requests.get(audio_url, timeout=60)
                if audio_file_resp.status_code == 200:
                    return audio_file_resp.content
        return None
    except Exception:
        return None


@app.post("/api/generateSong")
async def generate_song(payload: Dict[str, str]):
    """Generate a song using lyrics, genre, vocal style and duration.

    This function first attempts to call Stable Audio when lyrics are
    available.  If that fails or no API key is configured, it falls back
    to generating a simple sine-wave melody.
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
        audio_bytes = generate_stable_audio(prompt, duration)
    if audio_bytes:
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tune_{genre.lower()}_{int(duration)}s.wav"
            },
        )

    # Fallback: generate a simple melody if Stable Audio fails
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
    """Simple health check endpoint."""
    return {"status": "ok"}