from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List
import numpy as np
import random
import io
import wave

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
    prompt = payload.get("prompt", "")
    genre = payload.get("genre", "Unknown")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    lyrics = create_lyrics(prompt, genre)
    return JSONResponse(content={"lyrics": lyrics})

def create_lyrics(prompt: str, genre: str) -> str:
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
        "latin": ["amor", "ritmo", "baila", "corazon", "pasión"],
    }
    fillers = filler_by_genre.get(genre.lower(), ["story", "melody", "rhythm", "feeling"])
    lines = []
    for _ in range(4):
        line_words = []
        length = rng.randint(4, 8)
        for __ in range(length):
            if rng.random() < 0.3:
                line_words.append(rng.choice(fillers))
            else:
                line_words.append(rng.choice(words))
        if line_words:
            line_words[0] = line_words[0].capitalize()
        lines.append(" ".join(line_words))
    return "\n".join(lines)

@app.post("/api/generateSong")
async def generate_song(payload: Dict[str, str]):
    genre = payload.get("genre", "Unknown")
    duration = payload.get("duration")
    if duration is None:
        raise HTTPException(status_code=400, detail="Duration is required")
    try:
        duration = float(duration)
    except ValueError:
        raise HTTPException(status_code=400, detail="Duration must be a number")
    sample_rate = 44100
    audio_array = generate_melody(duration, genre, sample_rate)
    wav_bytes = to_wav_bytes(audio_array, sample_rate)
    filename = f"tune_{genre.lower()}_{int(duration)}s.wav"
    return StreamingResponse(
        wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

def generate_melody(duration: float, genre: str, sample_rate: int = 44100) -> np.ndarray:
    freqs_by_genre = {
        "pop": [261.63, 329.63, 392.00, 523.25],
        "rock": [82.41, 110.00, 146.83, 196.00],
        "hip-hop": [98.00, 123.47, 164.81, 196.00],
        "country": [130.81, 164.81, 196.00, 261.63],
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
        t = np.linspace(0, (end - start) / sample_rate, end - start, False)
        note = 0.5 * np.sin(2 * np.pi * freq * t)
        envelope = np.linspace(0, 1, len(note)) * np.linspace(1, 0, len(note))
        samples[start:end] = note * envelope
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples /= max_val
    return samples

def to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> io.BytesIO:
    audio_int16 = np.int16(audio_array * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return buf

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
