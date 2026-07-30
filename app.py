import os
import tempfile
import base64
import urllib.parse
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import demucs.separate
import demucs.pretrained

app = Flask(__name__)
CORS(app)

def fetch_lyrics(artist, title):
    try:
        url = f"https://api.lyrics.ovh/v1/{urllib.parse.quote(artist)}/{urllib.parse.quote(title)}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json().get('lyrics')
    except:
        pass
    return None

@app.route('/separate', methods=['POST'])
def separate():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = os.path.splitext(file.filename)[0]
    parts = filename.split(' - ')
    artist = parts[0].strip() if len(parts) >= 2 else None
    title = parts[1].strip() if len(parts) >= 2 else None
    lyrics = fetch_lyrics(artist, title) if artist and title else None

    with tempfile.TemporaryDirectory() as tmpdir:
        ext = file.filename.split(".")[-1]
        input_path = os.path.join(tmpdir, f"input.{ext}")
        file.save(input_path)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        try:
            demucs.separate.main(
                ["--model", "htdemucs", "--out", output_dir, input_path]
            )
        except Exception as e:
            return jsonify({"error": f"Demucs failed: {str(e)}"}), 500

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        sep_dir = os.path.join(output_dir, "htdemucs", base_name)
        vocals_path = os.path.join(sep_dir, "vocals.wav")
        inst_path = os.path.join(sep_dir, "no_vocals.wav")

        if not os.path.exists(vocals_path) or not os.path.exists(inst_path):
            return jsonify({"error": "Separation output missing"}), 500

        def encode_wav(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        return jsonify({
            "data": [{
                "instrumental": encode_wav(inst_path),
                "vocals": encode_wav(vocals_path),
                "lyrics": lyrics
            }]
        })

@app.route('/health')
def health():
    return {"status": "ok", "model": "htdemucs"}

@app.route('/model-status')
def model_status():
    try:
        # This will force a load if not cached, or return quickly if cached.
        model = demucs.pretrained.get_model('htdemucs')
        return {"cached": True}
    except Exception as e:
        return {"cached": False, "error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
