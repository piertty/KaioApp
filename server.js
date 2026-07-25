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

// Serve frontend files
app.use(express.static(path.join(__dirname, 'public')));

// Vocal separation endpoint
app.post('/api/separate', upload.single('audio'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No audio file uploaded' });
  }

  const audioBuffer = req.file.buffer;
  if (audioBuffer.length === 0) {
    return res.status(400).json({ error: 'Empty audio file' });
  }

  try {
    const apiUrl = 'https://api-inference.huggingface.co/models/facebook/htdemucs';

    // Retry logic for 503 (model cold start)
    const callWithRetry = async (retries = 3, delay = 8000) => {
      for (let i = 0; i < retries; i++) {
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${HF_API_TOKEN}`,
            'Content-Type': 'application/octet-stream',
          },
          body: audioBuffer,
        });

        if (response.status === 503) {
          console.log(`Model loading, retry ${i + 1}/${retries} after ${delay / 1000}s...`);
          await new Promise(r => setTimeout(r, delay));
          continue;
        }

        if (!response.ok) {
          const text = await response.text();
          throw new Error(`Hugging Face API error ${response.status}: ${text}`);
        }

        return response.json();
      }
      throw new Error('Model still loading after multiple retries. Please try again in a minute.');
    };

    const result = await callWithRetry(3, 8000);

    // Extract stems (handles both direct object and array responses)
    let vocals, other;
    if (result.vocals && result.other) {
      // Direct object
      vocals = result.vocals.data || result.vocals;
      other = result.other.data || result.other;
    } else if (Array.isArray(result) && result[0]) {
      // Array (some API versions)
      const stems = result[0];
      vocals = stems.vocals?.data || stems.vocals;
      other = stems.other?.data || stems.other;
    } else {
      return res.status(500).json({ error: 'Unexpected response format from API' });
    }

    if (!vocals || !other) {
      return res.status(500).json({ error: 'Could not extract stems from response' });
    }

    res.json({ vocals, other });
  } catch (error) {
    console.error('Separation error:', error.message);
    res.status(502).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Kaio server running on port ${PORT}`);
});