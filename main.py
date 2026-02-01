import base64
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from detector import classify_audio

app = FastAPI()

# Rule 5: API Authentication
VALID_API_KEY = "sk_test_123456789"

# Rule 7: Request Body Fields
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    # Rule 5 & 11: Validate API Key and handle error format
    if x_api_key != VALID_API_KEY:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

    # Rule 2: Supported Languages check
    supported = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if request.language not in supported:
        return {"status": "error", "message": "Unsupported language"}

    try:
        # Rule 4: Handle Base64 encoded MP3
        audio_bytes = base64.b64decode(request.audioBase64)
        
        # Analyze
        classification, score, reason = classify_audio(audio_bytes)

        # Rule 8 & 9: Final Success JSON Format
        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": score,
            "explanation": reason
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    