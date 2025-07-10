import os
import pickle
import uvicorn
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import face_recognition
from deepface import DeepFace

from pydantic import BaseModel, HttpUrl
import requests

class ImageURLRequest(BaseModel):
    image_url: HttpUrl

# ---------------------- Config ----------------------
UPLOADS_DIR = "uploads"
EMBEDDING_FILE = "user_embeddings.pkl"
MODEL_NAME = "Facenet"
DETECTOR = "retinaface"
SIMILARITY_THRESHOLD = 0.7

# --- Create necessary directories and files ---
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------------------- FastAPI App Setup ----------------------
app = FastAPI()
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
templates = Jinja2Templates(directory="templates")

# ---------------------- Helper Functions ----------------------
def load_embeddings():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            data = pickle.load(f)
        return pd.DataFrame(data)
    return pd.DataFrame(columns=["user_id", "embedding"])

def save_embeddings(df):
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(df.to_dict(orient="list"), f)

def get_embedding_from_bytes(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    # --- Face detection using face_recognition ---
    face_locations = face_recognition.face_locations(img_array)
    if not face_locations:
        raise ValueError("No face detected using face_recognition.")

    # --- Embedding generation using DeepFace ---
    try:
        embedding = DeepFace.represent(
            img_path=img_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False
        )[0]["embedding"]
        return embedding
    except Exception as e:
        raise ValueError(f"Embedding generation failed: {e}")

def find_duplicate(new_embedding, df):
    if df.empty:
        return {"duplicate": False}

    all_embeddings = np.array(df["embedding"].tolist())
    similarities = cosine_similarity([new_embedding], all_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            "duplicate": True,
            "matched_user_id": df.iloc[best_idx]["user_id"],
            "score": float(best_score),
        }
    return {"duplicate": False}

# ---------------------- API Endpoints ----------------------
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    response = templates.TemplateResponse("index.html", {"request": request})
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.post("/check-face")
async def check_face(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        new_embedding = get_embedding_from_bytes(image_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not process image or find face: {e}"}
        )

    df = load_embeddings()
    result = find_duplicate(new_embedding, df)

    if result.get("duplicate"):
        return JSONResponse(content={
            "status": "duplicate_found",
            "matched_image": result["matched_user_id"],
            "score": result["score"],
            "threshold": SIMILARITY_THRESHOLD
        })

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = os.path.join(UPLOADS_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    new_row = pd.DataFrame([{"user_id": filename, "embedding": new_embedding}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_embeddings(df)

    return JSONResponse(content={
        "status": "new_face_registered",
        "filename": filename,
        "threshold": SIMILARITY_THRESHOLD
    })

@app.post("/check-face-url")
async def check_face_url(request: ImageURLRequest):
    try:
        # Download the image from URL
        response = requests.get(request.image_url)
        if response.status_code != 200:
            return JSONResponse(status_code=400, content={"error": "Failed to download image from URL."})
        image_bytes = response.content

        # Generate embedding
        new_embedding = get_embedding_from_bytes(image_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not process image from URL: {e}"}
        )

    df = load_embeddings()
    result = find_duplicate(new_embedding, df)

    if result.get("duplicate"):
        return JSONResponse(content={
            "status": "duplicate_found",
            "matched_image": result["matched_user_id"],
            "score": result["score"],
            "threshold": SIMILARITY_THRESHOLD
        })

    # Save new image and embedding
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = os.path.join(UPLOADS_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    new_row = pd.DataFrame([{"user_id": filename, "embedding": new_embedding}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_embeddings(df)

    return JSONResponse(content={
        "status": "new_face_registered",
        "filename": filename,
        "threshold": SIMILARITY_THRESHOLD
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)

