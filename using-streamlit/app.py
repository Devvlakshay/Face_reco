import os
import pickle
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import streamlit as st
import uvicorn
import threading

# ---------------------- Config ----------------------
EMBEDDING_FILE = "user_embeddings.pkl"
MODEL_NAME = "Facenet"
DETECTOR = "retinaface"
SIMILARITY_THRESHOLD = 0.6

# ---------------------- Backend ----------------------
app = FastAPI()

# Allow Streamlit to access FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_embeddings():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            data = pickle.load(f)
        return pd.DataFrame(data)
    return pd.DataFrame(columns=["user_id", "embedding"])

def save_embeddings(df):
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(df.to_dict(orient="list"), f)

def get_embedding_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    if "image" not in response.headers.get("Content-Type", ""):
        raise Exception("URL does not point to a valid image.")

    img = Image.open(BytesIO(response.content)).convert("RGB")
    img_array = np.array(img)

    embedding = DeepFace.represent(
        img_path=img_array,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR
    )[0]["embedding"]

    return embedding

@app.post("/check-duplicate/")
async def check_duplicate(request: Request):
    body = await request.json()
    url = body.get("url")

    try:
        new_embedding = get_embedding_from_url(url)
    except Exception as e:
        return {"error": str(e)}

    df = load_embeddings()
    if not df.empty:
        all_embeddings = np.array(df["embedding"].tolist())
        similarities = cosine_similarity([new_embedding], all_embeddings)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= SIMILARITY_THRESHOLD:
            return {
                "duplicate": True,
                "matched_image": df.iloc[best_idx]["user_id"],
                "score": float(best_score),
                "threshold": SIMILARITY_THRESHOLD
            }

    # If not duplicate, add to database
    new_row = pd.DataFrame([{
        "user_id": url,
        "embedding": new_embedding
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_embeddings(df)

    return {
        "duplicate": False,
        "score": None,
        "threshold": SIMILARITY_THRESHOLD
    }

# ---------------------- Start Backend ----------------------
def start_backend():
    uvicorn.run(app, host="127.0.0.1", port=8000)

threading.Thread(target=start_backend, daemon=True).start()

# ---------------------- Streamlit UI ----------------------
st.title("🧠 Facial Duplicate Detector")
img_url = st.text_input("Enter an image URL:")

if img_url:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(img_url, headers=headers)
        response.raise_for_status()

        if "image" not in response.headers.get("Content-Type", ""):
            st.error("URL does not point to a valid image.")
        else:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(img, caption="Input Image")

            # Send to FastAPI
            res = requests.post("http://127.0.0.1:8000/check-duplicate/", json={"url": img_url})

            if res.status_code == 200:
                result = res.json()
                threshold = result.get( SIMILARITY_THRESHOLD)
                score = result.get("score")

                if result.get("duplicate"):
                    st.success(f"✅ Duplicate image found!\nMatched with: {result['matched_image']}")
                    st.info(f"Similarity Score: {score:.4f}")
                else:
                    st.success("✅ No duplicate found. Added to database.")
                    st.info(f"Threshold for future duplicates: {threshold}")
            else:
                st.error("❌ API returned an error.")
    except Exception as e:
        st.error(f"Error: {e}")
