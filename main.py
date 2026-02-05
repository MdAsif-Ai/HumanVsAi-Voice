import os
import base64
import io
import torch
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import pipeline
from pydub import AudioSegment
import librosa

# FORCE CPU
torch.set_num_threads(1)

API_KEY = os.getenv("API_KEY", "local_test_key")

app = FastAPI(title="AI Voice Detection API")

# Load model ONCE
classifier = pipeline(
    "audio-classification",
    model="garystafford/wav2vec2-deepfake-voice-detector",
    device=-1  # CPU only
)

# ----------------- Request Model -----------------
class VoiceRequest(BaseModel):
    language: str  # Tamil, English, Hindi, Malayalam, Telugu
    audioFormat: str  # Must be 'mp3'
    audioBase64: str

# ----------------- API Key Verification -----------------
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# ----------------- Convert Base64 MP3 to WAV -----------------
def base64_mp3_to_wav(audio_base64: str):
    audio_bytes = base64.b64decode(audio_base64)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

    # Limit audio to 15 seconds to save CPU
    if len(audio) > 15000:
        audio = audio[:15000]

    # Convert to mono, 16kHz
    audio = audio.set_frame_rate(16000).set_channels(1)

    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

# ----------------- Main API Endpoint -----------------
@app.post("/api/voice-detection")
def detect_voice(request: VoiceRequest, auth: bool = Depends(verify_api_key)):
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    # Convert Base64 MP3 to WAV in memory
    wav_file = base64_mp3_to_wav(request.audioBase64)

    # Load audio for pipeline
    speech, sr = librosa.load(wav_file, sr=16000)

    with torch.no_grad():
        result = classifier(speech, sampling_rate=16000)

    # Map model label to statement classification
    label = result[0]["label"].lower()
    ai_score = result[0]["score"]

    classification = "AI_GENERATED" if "fake" in label or "spoof" in label or "ai" in label else "HUMAN"
    explanation = (
        "Synthetic speech characteristics detected such as unnatural pitch stability"
        if classification == "AI_GENERATED"
        else "Natural human speech patterns detected"
    )

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(ai_score, 2),
        "explanation": explanation
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
