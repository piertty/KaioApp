const express = require('express');
const multer = require('multer');
const fetch = require('node-fetch');
const path = require('path');
require('dotenv').config();

const app = express();
const upload = multer();

const HF_API_TOKEN = process.env.HF_API_TOKEN;
if (!HF_API_TOKEN) {
  console.error('ERROR: HF_API_TOKEN environment variable is required.');
  process.exit(1);
}

app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/separate', upload.single('audio'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No audio file uploaded' });
  const audioBuffer = req.file.buffer;
  if (!audioBuffer.length) return res.status(400).json({ error: 'Empty file' });

  try {
    const apiUrl = 'https://api-inference.huggingface.co/models/facebook/htdemucs';

    // Retry up to 3 times on 503 (model loading)
    const callWithRetry = async (retries = 3, delay = 8000) => {
      for (let i = 0; i < retries; i++) {
        const resp = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${HF_API_TOKEN}`,
            'Content-Type': 'application/octet-stream',
          },
          body: audioBuffer,
        });
        if (resp.status === 503) {
          console.log(`Model loading, retry ${i + 1}/${retries}...`);
          await new Promise(r => setTimeout(r, delay));
          continue;
        }
        if (!resp.ok) throw new Error(`Hugging Face error ${resp.status}: ${await resp.text()}`);
        return resp.json();
      }
      throw new Error('Model still loading after retries. Please try again in a minute.');
    };

    const result = await callWithRetry(3, 8000);

    let vocals, other;
    if (result.vocals && result.other) {
      vocals = result.vocals.data || result.vocals;
      other = result.other.data || result.other;
    } else if (Array.isArray(result) && result[0]) {
      const stems = result[0];
      vocals = stems.vocals?.data || stems.vocals;
      other = stems.other?.data || stems.other;
    } else {
      return res.status(500).json({ error: 'Unexpected API response' });
    }

    if (!vocals || !other) return res.status(500).json({ error: 'Could not extract stems' });

    res.json({ vocals, other });
  } catch (err) {
    console.error(err);
    res.status(502).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Kaio running on port ${PORT}`));
