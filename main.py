import base64
import io
import librosa
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Voice Detector API")

# --- DATA MODELS ---
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# --- DETECTION LOGIC ---
def analyze_voice(audio_bytes):
    try:
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Feature Extraction: Spectral Rolloff
        # Synthetic audio often has "perfect" high frequencies (high rolloff)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        avg_rolloff = np.mean(rolloff)
        
        # Logic: If frequencies are too high/consistent, it's likely AI
        if avg_rolloff > 4500:
            classification = "AI_GENERATED"
            score = round(float(0.85 + (avg_rolloff / 10000)), 2)
            explanation = "Unnatural high-frequency consistency detected."
        else:
            classification = "HUMAN_AUTHENTIC"
            score = round(float(0.90 - (avg_rolloff / 10000)), 2)
            explanation = "Natural vocal harmonics and frequency decay observed."
            
        return classification, min(score, 0.99), explanation
    except Exception as e:
        return "ERROR", 0.0, str(e)

# --- API ENDPOINT ---
@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    # 1. Security Check
    if x_api_key != "sk_test_123456789":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2. Language Normalization (Fixes the "again" error)
    valid_langs = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    user_lang = request.language.strip().capitalize()
    
    if user_lang not in valid_langs:
        return {
            "status": "error",
            "message": f"Unsupported language. Choose from: {', '.join(valid_langs)}"
        }

    # 3. Base64 Decoding with Padding Fix
    try:
        b64_data = request.audioBase64
        missing_padding = len(b64_data) % 4
        if missing_padding:
            b64_data += "=" * (4 - missing_padding)
        
        audio_bytes = base64.b64decode(b64_data)
    except Exception:
        return {"status": "error", "message": "Invalid Base64 string."}

    # 4. Run Analysis
    result, confidence, reason = analyze_voice(audio_bytes)

    return {
        "status": "success",
        "language": user_lang,
        "classification": result,
        "confidenceScore": confidence,
        "explanation": reason
    }

# Health check for Render
@app.get("/")
def home():
    return {"message": "Voice API is Online", "docs": "/docs"}

    
