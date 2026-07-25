import os
import tempfile
import base64
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import demucs.separate
import librosa

app = FastAPI()

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
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "File must be an audio file")

    with tempfile.TemporaryDirectory() as tmpdir:
        ext = file.filename.split(".")[-1]
        input_path = os.path.join(tmpdir, f"input.{ext}")
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Set sys.argv for demucs.separate.main()
        sys.argv = ['demucs', '--model', 'htdemucs_ft', '--out', output_dir, input_path]
        try:
            demucs.separate.main()
        except Exception as e:
            raise HTTPException(500, f"Demucs error: {str(e)}")

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        sep_dir = os.path.join(output_dir, "htdemucs_ft", base_name)

        vocals_path = os.path.join(sep_dir, "vocals.wav")
        instrumental_path = os.path.join(sep_dir, "no_vocals.wav")

        if not os.path.exists(vocals_path) or not os.path.exists(instrumental_path):
            raise HTTPException(500, "Separation output missing")

        def encode_wav(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

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
