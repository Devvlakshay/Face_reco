import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from tempfile import NamedTemporaryFile
import shutil

# -----------------------------
# Config
EMBEDDING_FILE = "embeddings.pkl"
MODEL_NAME = "Facenet"
DETECTOR = "retinaface"
SIMILARITY_THRESHOLD = 0.8
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = FastAPI()
# -----------------------------

def load_embeddings():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            data = pickle.load(f)
        return pd.DataFrame(data)
    else:
        return pd.DataFrame(columns=["user_id", "embedding"])

def save_embeddings(df):
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(df.to_dict(orient="list"), f)

def extract_embedding(img_path):
    try:
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR
        )[0]["embedding"]
        return embedding
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

# ---------------
# API 1: Save embedding
@app.post("/store-embedding/")
async def store_embedding(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        temp_file = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract embedding
        embedding = extract_embedding(temp_file)

        # Load & update database
        df = load_embeddings()
        new_row = pd.DataFrame([{
            "user_id": file.filename,
            "embedding": embedding
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        save_embeddings(df)

        return {"message": "✅ Embedding stored successfully."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------
# API 2: Check for duplicates
@app.post("/check-duplicate/")
async def check_duplicate(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        temp_file = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract embedding
        new_embedding = extract_embedding(temp_file)

        # Load existing embeddings
        df = load_embeddings()

        if df.empty:
            # No previous data, store it directly
            new_row = pd.DataFrame([{
                "user_id": file.filename,
                "embedding": new_embedding
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            save_embeddings(df)
            return {"message": "📦 No prior data. Embedding stored as first entry."}

        all_embeddings = np.array(df["embedding"].tolist())
        similarities = cosine_similarity([new_embedding], all_embeddings)[0]

        top_idx = np.argmax(similarities)
        top_score = similarities[top_idx]

        if top_score >= SIMILARITY_THRESHOLD:
            duplicate_image = df.iloc[top_idx]["user_id"]
            return {
                "message": "⚠️ Duplicate Image Found",
                "matched_image": duplicate_image,
                "similarity": round(top_score, 4)
            }

        # If not duplicate, save it
        new_row = pd.DataFrame([{
            "user_id": file.filename,
            "embedding": new_embedding
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        save_embeddings(df)

        return {"message": "✅ No duplicate found. Embedding added to database."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

