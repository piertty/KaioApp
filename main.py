import io
import os
import json
import base64
import tempfile
import asyncio
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from demucs.pretrained import get_model
from demucs.apply import apply_model

MODEL_NAME = os.environ.get("DEMUCS_MODEL", "htdemucs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_ORIGINS = ["*"]
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30MB

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_error = None

@app.on_event("startup")
def load_model():
    global _model, _model_error
    try:
        print(f"[startup] loading Demucs model '{MODEL_NAME}' on {DEVICE} ...", flush=True)
        _model = get_model(MODEL_NAME)
        _model.to(DEVICE)
        _model.eval()
        print("[startup] model loaded OK.", flush=True)
    except Exception as e:
        _model_error = str(e)
        print(f"[startup] MODEL LOAD FAILED: {e}", flush=True)

@app.get("/health")
async def health():
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "version": "kaio-backend-v4-simple",
        "model": MODEL_NAME,
        "device": DEVICE,
        "model_loaded": _model is not None,
        "model_error": _model_error,
    }

def _run_separation(content: bytes, filename: str) -> dict:
    suffix = os.path.splitext(filename or "input")[1] or ".mp3"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        wav, sr = torchaudio.load(tmp_path)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        if sr != _model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, _model.samplerate)
            sr = _model.samplerate

        wav_batch = wav.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            sources = apply_model(_model, wav_batch, device=DEVICE, progress=False)[0]

        stem_names = list(_model.sources)
        vocals_idx = stem_names.index("vocals")

        vocals = sources[vocals_idx].cpu()
        instrumental = (sources.sum(dim=0) - sources[vocals_idx]).cpu()

        def to_wav_base64(tensor: torch.Tensor) -> str:
            buf = io.BytesIO()
            torchaudio.save(buf, tensor, sr, format="wav")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "vocals": to_wav_base64(vocals),
            "instrumental": to_wav_base64(instrumental),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, f"Model not loaded ({_model_error or 'unknown'})")
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(400, "File must be an audio file")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File too large – max {MAX_UPLOAD_BYTES // (1024*1024)}MB")

    # Run separation in a thread to avoid blocking the event loop
    try:
        result = await asyncio.to_thread(_run_separation, content, file.filename or "input")
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Separation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
