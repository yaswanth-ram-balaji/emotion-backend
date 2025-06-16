from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import sys
import traceback

app = FastAPI()

@app.middleware("http")
async def catch_exceptions_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print("GLOBAL ERROR CAUGHT BY MIDDLEWARE:", file=sys.stderr)
        traceback.print_exc()
        raise e

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your frontend domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageInput(BaseModel):
    image_base64: str
    method: str = "deepface"  # Changed default from "fer" to "deepface"

@app.post("/detect-face")
async def detect_face(payload: ImageInput):
    try:
        imgdata = base64.b64decode(payload.image_base64.split(",")[-1])
        img = Image.open(BytesIO(imgdata)).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        face_crop_base64 = base64.b64encode(buffered.getvalue()).decode()
        return {"face_crop_base64": face_crop_base64}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to process image: {e}")

@app.post("/analyze_emotion")
async def analyze_emotion(payload: ImageInput):
    # Force DeepFace method to avoid FER issues
    method = "deepface"  # Override whatever method is sent
    print(f"[DEBUG] Starting analyze_emotion with method: {method} (forced)", file=sys.stderr)
    
    try:
        imgdata = base64.b64decode(payload.image_base64.split(",")[-1])
        img = Image.open(BytesIO(imgdata)).convert("RGB")
        np_img = np.array(img)
        print(f"[DEBUG] Image processed successfully, shape: {np_img.shape}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Invalid image")

    # Always use DeepFace
    print("[DEBUG] Using DeepFace method", file=sys.stderr)
    try:
        from deepface import DeepFace
        print("[DEBUG] DeepFace imported successfully", file=sys.stderr)
        res = DeepFace.analyze(img_path=np_img, actions=['emotion'], enforce_detection=False)
        print(f"[DEBUG] DeepFace result: {res}", file=sys.stderr)
        
        # Fix: Handle the fact that res is a list, not a dict
        if res and len(res) > 0:
            first_face = res[0]  # Get the first detected face
            emotion = first_face['dominant_emotion']
            emotion_scores = first_face['emotion']
            face_confidence = first_face.get('face_confidence', 1.0)
            
            # Calculate confidence score (DeepFace returns percentages)
            score = float(emotion_scores[emotion]) / 100
            
            print(f"[DEBUG] Detected emotion: {emotion}, confidence: {score}", file=sys.stderr)
            
            return {
                "emotion": emotion, 
                "confidence": score,
                "emotion_scores": emotion_scores,
                "face_confidence": face_confidence
            }
        else:
            print("[ERROR] No faces detected in the image", file=sys.stderr)
            return {"emotion": "neutral", "confidence": 0.0, "error": "No faces detected"}
            
    except ImportError as e:
        print(f"[ERROR] DeepFace ImportError: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="DeepFace not installed")
    except Exception as e:
        print(f"[ERROR] DeepFace Exception: {e}", file=sys.stderr)
        traceback.print_exc()
        return {"emotion": "neutral", "confidence": 0.0, "error": str(e)}

@app.get("/")
async def root():
    return {"message": "Welcome to the Face Detection and Emotion Analysis API. Use /detect-face to detect faces and /analyze_emotion to analyze emotions from images."}