TuneCrafter Backend
===================

This directory contains a very simple FastAPI backend that mirrors the
functionality of Suno‑style music generation.  It exposes two endpoints:

* POST /api/generateLyrics – accepts a JSON body with a prompt and
  optional genre string and returns a JSON object with a lyrics
  field.  The implementation here uses a tiny, deterministic
  pseudo‑lyric generator.  For production you should replace the
  create_lyrics() function in main.py with a call to a large
  language model (e.g. GPT‑4, Llama‑3) running on your own servers.

* POST /api/generateSong – accepts a JSON body with lyrics,
  genre, vocal, and duration.  It attempts to generate an
  audio waveform from the given parameters.  In this skeleton, the
  music is procedurally generated using sine waves at genre‑dependent
  frequencies.  In a real deployment you should replace
  generate_melody() with a call to a transformer or diffusion model
  such as MusicGen or Stable Audio.  If you also generate a vocal
  track (e.g. with Bark), you can mix the two signals before
  returning them.

To run the server locally:

```
python -m pip install -r requirements.txt
uvicorn tune_crafter_backend.main:app --reload
```

You can then access the API endpoints at http://localhost:8000/api/*.

If you deploy this backend, update your frontend (e.g. the
`tunecrafter.html` file) to call these endpoints instead of the
placeholder functions.  The current HTML makes a `fetch()` call to
`/api/generateLyrics` and `/api/generateSong` if available; if the
request fails it falls back to local generation.
