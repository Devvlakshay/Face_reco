import os
import pickle
from io import BytesIO
from datetime import datetime
import logging
import psutil
import platform

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import requests

from deepface import DeepFace
import face_recognition

# Try to import CUDA-related libraries
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0
    CUDA_DEVICE_NAME = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Config ----------------------
EMBEDDING_FILE = "user_embeddings.pkl"
MODEL_NAME = "Facenet"
DETECTOR = "retinaface"
SIMILARITY_THRESHOLD = 0.7

# CUDA Configuration
if CUDA_AVAILABLE:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    logger.info(f"CUDA is available. Using GPU: {CUDA_DEVICE_NAME}")
else:
    logger.info("CUDA not available. Using CPU for processing.")

# ---------------------- FastAPI App ----------------------
app = FastAPI(
    title="Face Recognition API",
    description="Face recognition API with CUDA support for duplicate detection",
    version="1.0.0"
)

# ---------------------- Pydantic Schema ----------------------
class ImageURLRequest(BaseModel):
    user_id: str
    image_url: HttpUrl

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str = None
    cpu_count: int
    memory_usage: dict
    system_info: dict

# ---------------------- Helper Functions ----------------------
def get_system_info():
    """Get system information for health check"""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
    }

def get_memory_info():
    """Get memory usage information"""
    memory = psutil.virtual_memory()
    return {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "used_gb": round(memory.used / (1024**3), 2),
        "percent": memory.percent
    }

def load_embeddings():
    if os.path.exists(EMBEDDING_FILE):
        try:
            with open(EMBEDDING_FILE, "rb") as f:
                data = pickle.load(f)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return pd.DataFrame(columns=["user_id", "embedding"])
    return pd.DataFrame(columns=["user_id", "embedding"])

def save_embeddings(df):
    try:
        with open(EMBEDDING_FILE, "wb") as f:
            pickle.dump(df.to_dict(orient="list"), f)
        logger.info("Embeddings saved successfully")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        raise

def get_embedding_from_bytes(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        raise ValueError(f"Invalid image format: {e}")

    # Use face_recognition for face detection (works on both CPU and GPU)
    face_locations = face_recognition.face_locations(img_array)
    if not face_locations:
        raise ValueError("no_face_detected")

    try:
        # DeepFace will automatically use CUDA if available
        embedding = DeepFace.represent(
            img_path=img_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False
        )[0]["embedding"]
        
        logger.info(f"Embedding generated using {'GPU' if CUDA_AVAILABLE else 'CPU'}")
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise ValueError(f"embedding_generation_failed: {e}")

def find_duplicate(new_embedding, df):
    if df.empty:
        return {"duplicate": False}

    all_embeddings = np.array(df["embedding"].tolist())
    
    # Use GPU for cosine similarity if available
    if CUDA_AVAILABLE:
        try:
            import torch
            new_emb_tensor = torch.tensor([new_embedding]).cuda()
            all_emb_tensor = torch.tensor(all_embeddings).cuda()
            
            # Compute cosine similarity on GPU
            similarities = torch.nn.functional.cosine_similarity(
                new_emb_tensor.unsqueeze(1), 
                all_emb_tensor.unsqueeze(0), 
                dim=2
            ).cpu().numpy()[0]
            
            logger.info("Similarity computation performed on GPU")
        except Exception as e:
            logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            similarities = cosine_similarity([new_embedding], all_embeddings)[0]
    else:
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

# ---------------------- Health Endpoint ----------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system information"""
    try:
        # Test if we can load embeddings
        df = load_embeddings()
        total_embeddings = len(df)
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            cuda_available=CUDA_AVAILABLE,
            cuda_device_count=CUDA_DEVICE_COUNT,
            cuda_device_name=CUDA_DEVICE_NAME,
            cpu_count=psutil.cpu_count(),
            memory_usage=get_memory_info(),
            system_info={
                **get_system_info(),
                "total_registered_faces": total_embeddings,
                "embedding_file_exists": os.path.exists(EMBEDDING_FILE)
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "cuda_available": CUDA_AVAILABLE,
                "cuda_device_count": CUDA_DEVICE_COUNT
            }
        )

# ---------------------- Info Endpoint ----------------------
@app.get("/info")
async def get_info():
    """Get API configuration and capabilities"""
    return {
        "model_name": MODEL_NAME,
        "detector": DETECTOR,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "cuda_support": CUDA_AVAILABLE,
        "cuda_devices": CUDA_DEVICE_COUNT,
        "gpu_name": CUDA_DEVICE_NAME,
        "processing_device": "GPU" if CUDA_AVAILABLE else "CPU"
    }

# ---------------------- Main Endpoint ----------------------
@app.post("/check-face-url")
async def check_face_url(request: ImageURLRequest):
    start_time = datetime.now()
    
    # Step 1: Download image
    try:
        logger.info(f"Downloading image from URL for user: {request.user_id}")
        response = requests.get(request.image_url, timeout=30)
        if response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={"status": "failed_to_download_image_from_url"}
            )
        image_bytes = response.content
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "failed_to_download_image_from_url"}
        )

    # Step 2: Generate embedding
    try:
        logger.info("Generating face embedding...")
        new_embedding = get_embedding_from_bytes(image_bytes)
    except ValueError as e:
        error_status = str(e)
        if error_status == "no_face_detected":
            return JSONResponse(
                status_code=400,
                content={"status": "no_face_detected"}
            )
        elif "embedding_generation_failed" in error_status:
            return JSONResponse(
                status_code=400,
                content={"status": "embedding_generation_failed", "error": error_status}
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"status": "error_processing_image", "error": error_status}
            )

    # Step 3: Check for duplicates
    logger.info("Checking for duplicate faces...")
    df = load_embeddings()
    result = find_duplicate(new_embedding, df)

    processing_time = (datetime.now() - start_time).total_seconds()

    if result.get("duplicate"):
        logger.info(f"Duplicate face found for user: {request.user_id}")
        return JSONResponse(content={
            "status": "duplicate_image_found",
            "matched_user_id": result["matched_user_id"],
            "score": result["score"],
            # "threshold": SIMILARITY_THRESHOLD,
            # "processing_time_seconds": processing_time,
            # "processed_on": "GPU" if CUDA_AVAILABLE else "CPU"
        })

    # Step 4: Save new embedding
    logger.info(f"Registering new face for user: {request.user_id}")
    new_row = pd.DataFrame([{"user_id": request.user_id, "embedding": new_embedding}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_embeddings(df)

    return JSONResponse(content={
        "status": "new_face_registered",
        "user_id": request.user_id,
        # "threshold": SIMILARITY_THRESHOLD,
        # "processing_time_seconds": processing_time,
        # "processed_on": "GPU" if CUDA_AVAILABLE else "CPU",
        # "total_registered_faces": len(df)
    })

# ---------------------- Startup Event ----------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Face Recognition API starting up...")
    logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        logger.info(f"CUDA Devices: {CUDA_DEVICE_COUNT}")
        logger.info(f"GPU: {CUDA_DEVICE_NAME}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Detector: {DETECTOR}")
    logger.info(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")

# ---------------------- Run ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8100, 
        reload=True,
        log_level="info"
    )