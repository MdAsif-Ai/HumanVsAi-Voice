import os
import io
import base64
import torch
import librosa
import numpy as np

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import pipeline

# ---------------- CONFIG ----------------
torch.set_num_threads(1)
API_KEY = os.getenv("API_KEY", "local_test_key")

app = FastAPI(title="AI Voice Detection API")

# ---------------- LOAD MODEL ONCE ----------------
classifier = pipeline(
    task="audio-classification",
    model="garystafford/wav2vec2-deepfake-voice-detector",
    device=-1  # CPU only
)

# ---------------- REQUEST MODEL ----------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str  # wav or mp3 (wav strongly recommended)
    audioBase64: str

# ---------------- API KEY AUTH ----------------
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# ---------------- AUDIO DECODER (NO FFMPEG) ----------------
def decode_audio(audio_base64: str):
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)

        # librosa handles wav reliably, mp3 best-effort
        speech, sr = librosa.load(audio_buffer, sr=16000, mono=True)

        # Limit to 15 seconds
        max_len = 16000 * 15
        speech = speech[:max_len]

        if speech.size == 0:
            raise ValueError("Empty audio")

        return speech

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding failed: {str(e)}")

# ---------------- MAIN ENDPOINT ----------------
@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    auth: bool = Depends(verify_api_key)
):
    if request.audioFormat.lower() not in {"wav", "mp3"}:
        raise HTTPException(status_code=400, detail="audioFormat must be wav or mp3")

    speech = decode_audio(request.audioBase64)

    with torch.no_grad():
        result = classifier(speech, sampling_rate=16000)

    label = result[0]["label"].lower()
    score = float(result[0]["score"])

    classification = (
        "AI_GENERATED"
        if any(x in label for x in ["fake", "spoof", "ai"])
        else "HUMAN"
    )

    explanation = (
        "Synthetic speech characteristics detected"
        if classification == "AI_GENERATED"
        else "Natural human speech characteristics detected"
    )

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(score, 2),
        "model": "wav2vec2-deepfake-voice-detector",
        "explanation": explanation
    }

# ---------------- ENTRYPOINT ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
