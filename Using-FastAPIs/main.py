# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# import requests
# import numpy as np
# from io import BytesIO
# from PIL import Image, UnidentifiedImageError
# from deepface import DeepFace
# from sklearn.metrics.pairwise import cosine_similarity
# import redis
# import hashlib
# import os
# import logging
# from dotenv import load_dotenv

# # ---------------------- Load Config ----------------------
# load_dotenv()

# MODEL_NAME = "Facenet"
# DETECTOR = "retinaface"
# SIMILARITY_THRESHOLD = 0.6




# # ---------------------- Redis Setup ----------------------
# r = redis.Redis(
#     host=os.getenv("REDIS_HOST", "localhost"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )

# # ---------------------- Logging ----------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ---------------------- FastAPI App ----------------------
# app = FastAPI(title="Face Duplicate Checker API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------------------- Utility Functions ----------------------
# def get_embedding_from_url(url: str) -> np.ndarray:
#     try:
#         headers = {"User-Agent": "Mozilla/5.0"}
#         response = requests.get(url, headers=headers)
#         response.raise_for_status()

#         img = Image.open(BytesIO(response.content)).convert("RGB")
#         img_array = np.array(img)

#         embedding = DeepFace.represent(
#             img_path=img_array,
#             model_name=MODEL_NAME,
#             detector_backend=DETECTOR
#         )[0]["embedding"]

#         return np.array(embedding)
#     except UnidentifiedImageError:
#         logger.error("Cannot identify image file.")
#         raise Exception("Cannot identify image file.")
#     except Exception as e:
#         logger.error(f"Error processing image from URL: {str(e)}")
#         raise Exception(f"Error downloading or processing image: {str(e)}")

# def get_all_embeddings_from_redis():
#     keys = r.keys("face_embedding:*")
#     all_embeddings = []
#     for key in keys:
#         embedding_str = r.hget(key, "embedding")
#         if embedding_str:
#             try:
#                 embedding = np.fromiter(map(float, embedding_str.split(',')), dtype=float)
#                 all_embeddings.append((key, embedding))
#             except Exception as e:
#                 logger.warning(f"Failed to parse embedding for key {key}: {str(e)}")
#                 continue
#     return all_embeddings

# def store_embedding_in_redis(url: str, embedding: np.ndarray):
#     embedding_str = ','.join(map(str, embedding))
#     key_hash = hashlib.md5(url.encode()).hexdigest()
#     user_key = f"face_embedding:{key_hash}"
#     r.hset(user_key, mapping={"embedding": embedding_str})

# # ---------------------- API Endpoint ----------------------
# @app.post("/check-duplicate/")
# async def check_duplicate(request: Request):
#     try:
#         data = await request.json()
#         url = data.get("url")
#         if not url:
#             return {"error": "Image URL not provided."}

#         new_embedding = get_embedding_from_url(url)
#     except Exception as e:
#         return {"error": str(e)}

#     existing_embeddings = get_all_embeddings_from_redis()
#     if existing_embeddings:
#         stored_vectors = [e[1] for e in existing_embeddings]
#         similarities = cosine_similarity([new_embedding], stored_vectors)[0]
#         best_idx = int(np.argmax(similarities))
#         best_score = float(similarities[best_idx])

#         if best_score >= SIMILARITY_THRESHOLD:
#             return {
#                 "result": "Duplicate image found",
#                 "similarity_score": best_score,
#                 "threshold": SIMILARITY_THRESHOLD,
#                 "matched_key": existing_embeddings[best_idx][0]
#             }

#     store_embedding_in_redis(url, new_embedding)

#     return {
#         "result": "No duplicate found",
#         "similarity_score": None,
#         "threshold": SIMILARITY_THRESHOLD
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8100, reload=True)

# main.py

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

from deepface import DeepFace

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
    embedding = DeepFace.represent(
        img_path=img_array,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True
    )[0]["embedding"]
    return embedding

def find_duplicate(new_embedding, df):
    if df.empty:
        return {"duplicate": False}
        
    # ## DEBUG ##: Convert embeddings to a format suitable for cosine similarity
    # This is a critical step. df['embedding'].tolist() should give a list of lists.
    all_embeddings = np.array(df["embedding"].tolist())
    
    # ## DEBUG ##: Print the shapes to ensure they are compatible
    print(f"## DEBUG ##: Shape of new_embedding: {np.array(new_embedding).shape}")
    print(f"## DEBUG ##: Shape of all_embeddings from file: {all_embeddings.shape}")
    
    # Calculate similarities
    similarities = cosine_similarity([new_embedding], all_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    print(f"## DEBUG ##: Best similarity score found: {best_score}")

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
    """Serve the main HTML page with the camera interface."""
    response = templates.TemplateResponse("index.html", {"request": request})
    # Add headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.post("/check-face/")
async def check_face(file: UploadFile = File(...)):
    print("\n--- Received New Request ---")
    try:
        image_bytes = await file.read()
        new_embedding = get_embedding_from_bytes(image_bytes)
        print("## DEBUG ##: Successfully generated new embedding.")
    except Exception as e:
        print(f"## ERROR ##: Could not process image: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not process image or find face: {e}"}
        )

    df = load_embeddings()
    print(f"## DEBUG ##: Loaded embeddings DataFrame. Shape: {df.shape}")
    if not df.empty:
        print(df.head())

    result = find_duplicate(new_embedding, df)
    print(f"## DEBUG ##: Result from find_duplicate function: {result}")

    if result.get("duplicate"):
        return JSONResponse(content={
            "status": "duplicate_found",
            "matched_image": result["matched_user_id"],
            "score": result["score"],
            "threshold": SIMILARITY_THRESHOLD
        })

    # --- If not a duplicate, save the image and the embedding ---
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = os.path.join(UPLOADS_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(image_bytes)
    print(f"## DEBUG ##: New image saved to {save_path}")

    new_row = pd.DataFrame([{"user_id": filename, "embedding": new_embedding}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_embeddings(df)
    print("## DEBUG ##: New embedding saved to pickle file.")

    return JSONResponse(content={
        "status": "new_face_registered",
        "filename": filename,
        "threshold": SIMILARITY_THRESHOLD
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)