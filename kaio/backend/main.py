import os
import tempfile
import base64
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import demucs.separate
import librosa

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "File must be an audio file")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded file
        input_path = os.path.join(tmpdir, "input." + file.filename.split(".")[-1])
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Run Demucs separation (htdemucs_ft model)
        try:
            demucs.separate.main(
                ["--model", "htdemucs_ft", "--out", output_dir, input_path]
            )
        except Exception as e:
            raise HTTPException(500, f"Demucs failed: {str(e)}")

        # Locate the separated files
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        sep_dir = os.path.join(output_dir, "htdemucs_ft", base_name)

        vocals_path = os.path.join(sep_dir, "vocals.wav")
        instrumental_path = os.path.join(sep_dir, "no_vocals.wav")

        if not os.path.exists(vocals_path) or not os.path.exists(instrumental_path):
            raise HTTPException(500, "Separation output missing")

        # Helper to encode WAV to base64
        def encode_wav(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # Get duration for UI
        def get_info(path):
            y, sr = librosa.load(path, sr=None)
            return {"duration": len(y) / sr, "sample_rate": sr}

        return JSONResponse({
            "instrumental": encode_wav(instrumental_path),
            "vocals": encode_wav(vocals_path),
            "instrumental_info": get_info(instrumental_path),
            "vocals_info": get_info(vocals_path),
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)