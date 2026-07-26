import io
import os
import base64
import tempfile

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from demucs.pretrained import get_model
from demucs.apply import apply_model

# ── Config ───────────────────────────────────────────────────────────
# "htdemucs" (single model) instead of "htdemucs_ft" (4-model ensemble).
# The _ft ensemble is a little higher quality but needs ~4x the RAM and
# compute to load and run — on a Render Starter/Standard instance that's
# the difference between "works" and "gets OOM-killed mid-request".
# If you upgrade to a bigger instance (or add a GPU), you can switch this
# back via the DEMUCS_MODEL env var without touching code.
MODEL_NAME = os.environ.get("DEMUCS_MODEL", "htdemucs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lock this down to your actual Netlify URL once you know it, e.g.
# "https://kaio-studio.netlify.app". "*" works but is wide open — fine
# while you're testing, worth tightening once this is live.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # must be False when allow_origins includes "*" — the two are mutually exclusive per the CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_error = None


@app.on_event("startup")
def load_model():
    """Load the model ONCE when the server boots, not on every request.
    This is the single biggest fix here — the original code re-loaded
    htdemucs_ft from disk on every /separate call, which is both slow
    and a major source of memory churn/crashes under any real load."""
    global _model, _model_error
    try:
        print(f"[startup] loading Demucs model '{MODEL_NAME}' on {DEVICE} ...", flush=True)
        _model = get_model(MODEL_NAME)
        _model.to(DEVICE)
        _model.eval()
        print("[startup] model loaded OK.", flush=True)
    except Exception as e:
        # Don't let a bad model load silently crash-loop the whole process —
        # surface it clearly through /health instead so it's debuggable.
        _model_error = str(e)
        print(f"[startup] MODEL LOAD FAILED: {e}", flush=True)


@app.get("/health")
async def health():
    """Hit this directly in your browser first when debugging —
    https://your-service.onrender.com/health — before touching the
    frontend at all. If this doesn't return {"model_loaded": true},
    the frontend was never going to work regardless of BACKEND_URL."""
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "version": "kaio-backend-v2-startup-load",  # if you don't see this string, Render is still running old code
        "model": MODEL_NAME,
        "device": DEVICE,
        "model_loaded": _model is not None,
        "model_error": _model_error,
    }


@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, f"Model isn't loaded ({_model_error or 'still starting up'}). Check /health.")

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(400, "File must be an audio file")

    suffix = os.path.splitext(file.filename or "input")[1] or ".mp3"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        wav, sr = torchaudio.load(tmp_path)  # (channels, samples)

        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)  # demucs expects stereo in, stereo out
        elif wav.shape[0] > 2:
            wav = wav[:2]

        if sr != _model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, _model.samplerate)
            sr = _model.samplerate

        wav_batch = wav.unsqueeze(0).to(DEVICE)  # (1, channels, samples)

        with torch.no_grad():
            # apply_model handles chunking/overlap-add internally for
            # long songs — no need to hand-roll that ourselves.
            sources = apply_model(_model, wav_batch, device=DEVICE, progress=False)[0]
            # sources: (num_stems, channels, samples) — order given by _model.sources

        stem_names = list(_model.sources)  # typically ['drums', 'bass', 'other', 'vocals']
        vocals_idx = stem_names.index("vocals")

        vocals = sources[vocals_idx].cpu()
        # Instrumental = everything that isn't vocals, summed back together —
        # this is the actual "no_vocals" / karaoke track.
        instrumental = (sources.sum(dim=0) - sources[vocals_idx]).cpu()

        def to_wav_base64(tensor: torch.Tensor) -> str:
            buf = io.BytesIO()
            torchaudio.save(buf, tensor, sr, format="wav")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse({
            "vocals": to_wav_base64(vocals),
            "instrumental": to_wav_base64(instrumental),
            "sample_rate": sr,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Separation failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
