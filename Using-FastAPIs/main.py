# import os
# import pickle
# from io import BytesIO
# from datetime import datetime
# import logging
# import psutil
# import platform

# import numpy as np
# import pandas as pd
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity

# from fastapi import FastAPI
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, HttpUrl
# import requests

# from deepface import DeepFace
# import face_recognition

# # Try to import CUDA-related libraries
# try:
#     import torch
#     CUDA_AVAILABLE = torch.cuda.is_available()
#     CUDA_DEVICE_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
#     CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None
# except ImportError:
#     CUDA_AVAILABLE = False
#     CUDA_DEVICE_COUNT = 0
#     CUDA_DEVICE_NAME = None

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ---------------------- Config ----------------------
# EMBEDDING_FILE = "user_embeddings.pkl"
# MODEL_NAME = "Facenet"
# DETECTOR = "retinaface"
# SIMILARITY_THRESHOLD = 0.7

# # CUDA Configuration
# if CUDA_AVAILABLE:
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
#     logger.info(f"CUDA is available. Using GPU: {CUDA_DEVICE_NAME}")
# else:
#     logger.info("CUDA not available. Using CPU for processing.")

# # ---------------------- FastAPI App ----------------------
# app = FastAPI(
#     title="Face Recognition API",
#     description="Face recognition API with CUDA support for duplicate detection",
#     version="1.0.0"
# )

# # ---------------------- Pydantic Schema ----------------------
# class ImageURLRequest(BaseModel):
#     user_id: str
#     image_url: HttpUrl

# class HealthResponse(BaseModel):
#     status: str
#     timestamp: str
#     cuda_available: bool
#     cuda_device_count: int
#     cuda_device_name: str = None
#     cpu_count: int
#     memory_usage: dict
#     system_info: dict

# # ---------------------- Helper Functions ----------------------
# def get_system_info():
#     """Get system information for health check"""
#     return {
#         "platform": platform.platform(),
#         "processor": platform.processor(),
#         "architecture": platform.architecture()[0],
#         "python_version": platform.python_version(),
#         "cpu_count": psutil.cpu_count(),
#         "cpu_percent": psutil.cpu_percent(interval=1),
#     }

# def get_memory_info():
#     """Get memory usage information"""
#     memory = psutil.virtual_memory()
#     return {
#         "total_gb": round(memory.total / (1024**3), 2),
#         "available_gb": round(memory.available / (1024**3), 2),
#         "used_gb": round(memory.used / (1024**3), 2),
#         "percent": memory.percent
#     }

# def load_embeddings():
#     if os.path.exists(EMBEDDING_FILE):
#         try:
#             with open(EMBEDDING_FILE, "rb") as f:
#                 data = pickle.load(f)
#             return pd.DataFrame(data)
#         except Exception as e:
#             logger.error(f"Error loading embeddings: {e}")
#             return pd.DataFrame(columns=["user_id", "embedding"])
#     return pd.DataFrame(columns=["user_id", "embedding"])

# def save_embeddings(df):
#     try:
#         with open(EMBEDDING_FILE, "wb") as f:
#             pickle.dump(df.to_dict(orient="list"), f)
#         logger.info("Embeddings saved successfully")
#     except Exception as e:
#         logger.error(f"Error saving embeddings: {e}")
#         raise

# def get_embedding_from_bytes(image_bytes):
#     try:
#         img = Image.open(BytesIO(image_bytes)).convert("RGB")
#         img_array = np.array(img)
#     except Exception as e:
#         raise ValueError(f"Invalid image format: {e}")

#     # Use face_recognition for face detection (works on both CPU and GPU)
#     face_locations = face_recognition.face_locations(img_array)
#     if not face_locations:
#         raise ValueError("no_face_detected")

#     try:
#         # DeepFace will automatically use CUDA if available
#         embedding = DeepFace.represent(
#             img_path=img_array,
#             model_name=MODEL_NAME,
#             detector_backend=DETECTOR,
#             enforce_detection=False
#         )[0]["embedding"]
        
#         logger.info(f"Embedding generated using {'GPU' if CUDA_AVAILABLE else 'CPU'}")
#         return embedding
#     except Exception as e:
#         logger.error(f"Embedding generation failed: {e}")
#         raise ValueError(f"embedding_generation_failed: {e}")

# def find_duplicate(new_embedding, df):
#     if df.empty:
#         return {"duplicate": False}

#     all_embeddings = np.array(df["embedding"].tolist())
    
#     # Use GPU for cosine similarity if available
#     if CUDA_AVAILABLE:
#         try:
#             import torch
#             new_emb_tensor = torch.tensor([new_embedding]).cuda()
#             all_emb_tensor = torch.tensor(all_embeddings).cuda()
            
#             # Compute cosine similarity on GPU
#             similarities = torch.nn.functional.cosine_similarity(
#                 new_emb_tensor.unsqueeze(1), 
#                 all_emb_tensor.unsqueeze(0), 
#                 dim=2
#             ).cpu().numpy()[0]
            
#             logger.info("Similarity computation performed on GPU")
#         except Exception as e:
#             logger.warning(f"GPU computation failed, falling back to CPU: {e}")
#             similarities = cosine_similarity([new_embedding], all_embeddings)[0]
#     else:
#         similarities = cosine_similarity([new_embedding], all_embeddings)[0]
    
#     best_idx = np.argmax(similarities)
#     best_score = similarities[best_idx]

#     if best_score >= SIMILARITY_THRESHOLD:
#         return {
#             "duplicate": True,
#             "matched_user_id": df.iloc[best_idx]["user_id"],
#             "score": float(best_score),
#         }
#     return {"duplicate": False}

# # ---------------------- Health Endpoint ----------------------
# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint with system information"""
#     try:
#         # Test if we can load embeddings
#         df = load_embeddings()
#         total_embeddings = len(df)
        
#         return HealthResponse(
#             status="healthy",
#             timestamp=datetime.now().isoformat(),
#             cuda_available=CUDA_AVAILABLE,
#             cuda_device_count=CUDA_DEVICE_COUNT,
#             cuda_device_name=CUDA_DEVICE_NAME,
#             cpu_count=psutil.cpu_count(),
#             memory_usage=get_memory_info(),
#             system_info={
#                 **get_system_info(),
#                 "total_registered_faces": total_embeddings,
#                 "embedding_file_exists": os.path.exists(EMBEDDING_FILE)
#             }
#         )
#     except Exception as e:
#         logger.error(f"Health check failed: {e}")
#         return JSONResponse(
#             status_code=503,
#             content={
#                 "status": "unhealthy",
#                 "timestamp": datetime.now().isoformat(),
#                 "error": str(e),
#                 "cuda_available": CUDA_AVAILABLE,
#                 "cuda_device_count": CUDA_DEVICE_COUNT
#             }
#         )

# # ---------------------- Info Endpoint ----------------------
# @app.get("/info")
# async def get_info():
#     """Get API configuration and capabilities"""
#     return {
#         "model_name": MODEL_NAME,
#         "detector": DETECTOR,
#         "similarity_threshold": SIMILARITY_THRESHOLD,
#         "cuda_support": CUDA_AVAILABLE,
#         "cuda_devices": CUDA_DEVICE_COUNT,
#         "gpu_name": CUDA_DEVICE_NAME,
#         "processing_device": "GPU" if CUDA_AVAILABLE else "CPU"
#     }

# # ---------------------- Main Endpoint ----------------------
# @app.post("/check-face-url")
# async def check_face_url(request: ImageURLRequest):
#     start_time = datetime.now()
    
#     # Step 1: Download image
#     try:
#         logger.info(f"Downloading image from URL for user: {request.user_id}")
#         response = requests.get(request.image_url, timeout=30)
#         if response.status_code != 200:
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "failed_to_download_image_from_url"}
#             )
#         image_bytes = response.content
#     except Exception as e:
#         logger.error(f"Failed to download image: {e}")
#         return JSONResponse(
#             status_code=400,
#             content={"status": "failed_to_download_image_from_url"}
#         )

#     # Step 2: Generate embedding
#     try:
#         logger.info("Generating face embedding...")
#         new_embedding = get_embedding_from_bytes(image_bytes)
#     except ValueError as e:
#         error_status = str(e)
#         if error_status == "no_face_detected":
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "no_face_detected"}
#             )
#         elif "embedding_generation_failed" in error_status:
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "embedding_generation_failed", "error": error_status}
#             )
#         else:
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "error_processing_image", "error": error_status}
#             )

#     # Step 3: Check for duplicates
#     logger.info("Checking for duplicate faces...")
#     df = load_embeddings()
#     result = find_duplicate(new_embedding, df)

#     processing_time = (datetime.now() - start_time).total_seconds()

#     if result.get("duplicate"):
#         logger.info(f"Duplicate face found for user: {request.user_id}")
#         return JSONResponse(content={
#             "status": "duplicate_image_found",
#             "matched_user_id": result["matched_user_id"],
#             "score": result["score"],
#             # "threshold": SIMILARITY_THRESHOLD,
#             # "processing_time_seconds": processing_time,
#             # "processed_on": "GPU" if CUDA_AVAILABLE else "CPU"
#         })

#     # Step 4: Save new embedding
#     logger.info(f"Registering new face for user: {request.user_id}")
#     new_row = pd.DataFrame([{"user_id": request.user_id, "embedding": new_embedding}])
#     df = pd.concat([df, new_row], ignore_index=True)
#     save_embeddings(df)

#     return JSONResponse(content={
#         "status": "new_face_registered",
#         "user_id": request.user_id,
#         # "threshold": SIMILARITY_THRESHOLD,
#         # "processing_time_seconds": processing_time,
#         # "processed_on": "GPU" if CUDA_AVAILABLE else "CPU",
#         # "total_registered_faces": len(df)
#     })

# # ---------------------- Startup Event ----------------------
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Face Recognition API starting up...")
#     logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
#     if CUDA_AVAILABLE:
#         logger.info(f"CUDA Devices: {CUDA_DEVICE_COUNT}")
#         logger.info(f"GPU: {CUDA_DEVICE_NAME}")
#     logger.info(f"Model: {MODEL_NAME}")
#     logger.info(f"Detector: {DETECTOR}")
#     logger.info(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")

# # ---------------------- Run ----------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0", 
#         port=8100, 
#         reload=True,
#         log_level="info"
#     )


# # -------------------------------- HELLO ----------------------------------------------------
import os
import pickle
from io import BytesIO
from datetime import datetime
import logging
import psutil
import platform
import cv2

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import requests

from deepface import DeepFace

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
EMPLOYEE_EMBEDDING_FILE = "employee_embeddings.pkl"
MODEL_NAME = "Facenet"
DETECTOR = "retinaface"
SIMILARITY_THRESHOLD = 0.7
EMPLOYEE_SIMILARITY_THRESHOLD = 0.75

# CUDA Configuration
if CUDA_AVAILABLE:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logger.info(f"CUDA is available. Using GPU: {CUDA_DEVICE_NAME}")
else:
    logger.info("CUDA not available. Using CPU for processing.")

# ---------------------- FastAPI App ----------------------
app = FastAPI(
    title="Enhanced Face Recognition API",
    description="Face recognition API with CUDA support, multiple detection backends, and preprocessing",
    version="2.0.0"
)

# ---------------------- Pydantic Schema ----------------------
class ImageURLRequest(BaseModel):
    user_id: str
    image_url: HttpUrl

class EmployeeRegistrationRequest(BaseModel):
    employee_id: str
    employee_name: str
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

def load_employee_embeddings():
    """Load employee embeddings from pickle file"""
    if os.path.exists(EMPLOYEE_EMBEDDING_FILE):
        try:
            with open(EMPLOYEE_EMBEDDING_FILE, "rb") as f:
                data = pickle.load(f)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error loading employee embeddings: {e}")
            return pd.DataFrame(columns=["employee_id", "employee_name", "embedding"])
    return pd.DataFrame(columns=["employee_id", "employee_name", "embedding"])

def save_embeddings(df):
    try:
        with open(EMBEDDING_FILE, "wb") as f:
            pickle.dump(df.to_dict(orient="list"), f)
        logger.info("User embeddings saved successfully")
    except Exception as e:
        logger.error(f"Error saving user embeddings: {e}")
        raise

def save_employee_embeddings(df):
    """Save employee embeddings to pickle file"""
    try:
        with open(EMPLOYEE_EMBEDDING_FILE, "wb") as f:
            pickle.dump(df.to_dict(orient="list"), f)
        logger.info("Employee embeddings saved successfully")
    except Exception as e:
        logger.error(f"Error saving employee embeddings: {e}")
        raise

def preprocess_image_for_detection(img_array):
    """Apply various preprocessing techniques to improve face detection"""
    preprocessing_methods = []
    
    try:
        # Method 1: Histogram equalization for better contrast
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.equalizeHist(gray)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        preprocessing_methods.append(("histogram_equalization", enhanced_rgb))
    except Exception as e:
        logger.warning(f"Histogram equalization failed: {e}")
    
    try:
        # Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        preprocessing_methods.append(("clahe", clahe_rgb))
    except Exception as e:
        logger.warning(f"CLAHE preprocessing failed: {e}")
    
    try:
        # Method 3: Brightness and contrast adjustment using PIL
        pil_img = Image.fromarray(img_array)
        
        # Increase brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(pil_img)
        bright_img = brightness_enhancer.enhance(1.2)
        
        # Increase contrast
        contrast_enhancer = ImageEnhance.Contrast(bright_img)
        enhanced_img = contrast_enhancer.enhance(1.3)
        
        enhanced_array = np.array(enhanced_img)
        preprocessing_methods.append(("brightness_contrast", enhanced_array))
    except Exception as e:
        logger.warning(f"PIL enhancement failed: {e}")
    
    try:
        # Method 4: Gamma correction for lighting issues
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(img_array, table)
        preprocessing_methods.append(("gamma_correction", gamma_corrected))
    except Exception as e:
        logger.warning(f"Gamma correction failed: {e}")
    
    try:
        # Method 5: Gaussian blur to reduce noise, then sharpen
        blurred = cv2.GaussianBlur(img_array, (3, 3), 0)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        preprocessing_methods.append(("blur_sharpen", sharpened))
    except Exception as e:
        logger.warning(f"Blur-sharpen preprocessing failed: {e}")
    
    return preprocessing_methods

def validate_face_detection(img_array, detection_result):
    """
    Validate that the detected face is actually a human face and not a false positive
    """
    try:
        if not detection_result or len(detection_result) == 0:
            return False, "No detection result"
        
        # Get the face region details if available
        face_info = detection_result[0]
        
        # Check if we have facial area information
        if 'facial_area' in face_info:
            facial_area = face_info['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            # Extract face region
            face_region = img_array[y:y+h, x:x+w]
            
            # Validate face region size - too small faces are likely false positives
            if w < 80 or h < 80:
                return False, f"Face region too small: {w}x{h} (minimum 80x80)"
            
            # Check aspect ratio - human faces should have reasonable width/height ratio
            aspect_ratio = w / h
            if aspect_ratio < 0.6 or aspect_ratio > 1.8:
                return False, f"Invalid face aspect ratio: {aspect_ratio:.2f} (should be 0.6-1.8)"
            
            # Check if face takes up reasonable portion of image
            img_area = img_array.shape[0] * img_array.shape[1]
            face_area = w * h
            face_percentage = (face_area / img_area) * 100
            
            # Face should be at least 1% but not more than 80% of image
            if face_percentage < 1.0:
                return False, f"Face too small relative to image: {face_percentage:.1f}%"
            if face_percentage > 80.0:
                return False, f"Face too large relative to image: {face_percentage:.1f}%"
        
        # Additional validation using confidence scores if available
        if 'confidence' in face_info and face_info['confidence'] < 0.7:
            return False, f"Low confidence detection: {face_info['confidence']:.3f}"
        
        return True, "Valid face detected"
        
    except Exception as e:
        logger.warning(f"Face validation failed: {e}")
        # If validation fails, be conservative and reject
        return False, f"Validation error: {e}"

def get_embedding_from_bytes(image_bytes):
    """
    Enhanced face detection and embedding generation with strict validation
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        logger.info(f"Processing image: {width}x{height} pixels, size: {len(image_bytes)} bytes")
        
        # Check if image is too small
        if width < 100 or height < 100:
            logger.warning(f"Image might be too small for reliable face detection: {width}x{height}")
        
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise ValueError(f"Invalid image format: {e}")

    # Define detection backends to try in order of preference (most reliable first)
    detection_backends = [
        "retinaface",    # Most accurate, good for various angles
        "mtcnn",         # Good for small faces, reliable
        "dlib",          # Classical method, very reliable
        "opencv",        # Fast, good for frontal faces  
        "ssd",           # Good balance of speed and accuracy
        "mediapipe",     # Good for real-time processing
    ]
    
    # Track all attempts for logging
    detection_attempts = []
    successful_detection = None
    
    # Try original image first with all backends
    for backend in detection_backends:
        try:
            logger.info(f"Attempting detection with {backend} on original image")
            
            result = DeepFace.represent(
                img_path=img_array,
                model_name=MODEL_NAME,
                detector_backend=backend,
                enforce_detection=True
            )
            
            if result and len(result) > 0:
                # IMPORTANT: Validate the detection before accepting it
                is_valid, validation_msg = validate_face_detection(img_array, result)
                
                if is_valid:
                    embedding = result[0]["embedding"]
                    successful_detection = {
                        "method": f"{backend}_original",
                        "backend": backend,
                        "preprocessing": "none",
                        "embedding": embedding,
                        "faces_found": len(result),
                        "validation": validation_msg
                    }
                    logger.info(f"✓ Valid face detected using {backend} on original image")
                    logger.info(f"   Validation: {validation_msg}")
                    break
                else:
                    logger.warning(f"✗ {backend} detected face but validation failed: {validation_msg}")
                    detection_attempts.append({
                        "backend": backend,
                        "preprocessing": "none", 
                        "error": f"Validation failed: {validation_msg}",
                        "error_type": "invalid_face"
                    })
                    continue
                
        except Exception as e:
            error_msg = str(e).lower()
            detection_attempts.append({
                "backend": backend,
                "preprocessing": "none", 
                "error": str(e),
                "error_type": "no_face" if "face" in error_msg else "processing_error"
            })
            logger.debug(f"✗ {backend} failed on original image: {e}")
            continue
    
    # If original image failed, try with preprocessing (but with stricter validation)
    if not successful_detection:
        logger.info("Original image detection failed, trying preprocessing methods...")
        
        preprocessing_methods = preprocess_image_for_detection(img_array)
        
        # Limit preprocessing attempts to avoid too many false positives
        for prep_name, preprocessed_img in preprocessing_methods[:3]:  # Only try first 3 methods
            logger.info(f"Trying preprocessing method: {prep_name}")
            
            # Only try the most reliable backends with preprocessing
            reliable_backends = ["retinaface", "mtcnn", "dlib"]
            
            for backend in reliable_backends:
                try:
                    logger.debug(f"Attempting {backend} with {prep_name} preprocessing")
                    
                    result = DeepFace.represent(
                        img_path=preprocessed_img,
                        model_name=MODEL_NAME,
                        detector_backend=backend,
                        enforce_detection=True
                    )
                    
                    if result and len(result) > 0:
                        # IMPORTANT: Strict validation for preprocessed images
                        is_valid, validation_msg = validate_face_detection(preprocessed_img, result)
                        
                        if is_valid:
                            embedding = result[0]["embedding"]
                            successful_detection = {
                                "method": f"{backend}_{prep_name}",
                                "backend": backend,
                                "preprocessing": prep_name,
                                "embedding": embedding,
                                "faces_found": len(result),
                                "validation": validation_msg
                            }
                            logger.info(f"✓ Valid face detected using {backend} with {prep_name} preprocessing")
                            logger.info(f"   Validation: {validation_msg}")
                            break
                        else:
                            logger.warning(f"✗ {backend} with {prep_name} detected face but validation failed: {validation_msg}")
                            detection_attempts.append({
                                "backend": backend,
                                "preprocessing": prep_name,
                                "error": f"Validation failed: {validation_msg}",
                                "error_type": "invalid_face"
                            })
                            continue
                        
                except Exception as e:
                    error_msg = str(e).lower()
                    detection_attempts.append({
                        "backend": backend,
                        "preprocessing": prep_name,
                        "error": str(e),
                        "error_type": "no_face" if "face" in error_msg else "processing_error"
                    })
                    logger.debug(f"✗ {backend} with {prep_name} failed: {e}")
                    continue
            
            # If we found a valid face with this preprocessing, break out of preprocessing loop
            if successful_detection:
                break
    
    # Log detailed results
    if successful_detection:
        logger.info(f"🎉 SUCCESS: Face detection completed using {successful_detection['method']}")
        logger.info(f"   - Backend: {successful_detection['backend']}")
        logger.info(f"   - Preprocessing: {successful_detection['preprocessing']}")
        logger.info(f"   - Faces found: {successful_detection['faces_found']}")
        logger.info(f"   - Total attempts before success: {len(detection_attempts)}")
        logger.info(f"   - Processing device: {'GPU' if CUDA_AVAILABLE else 'CPU'}")
        
        return successful_detection["embedding"]
    
    else:
        # Comprehensive failure logging
        logger.error("❌ FACE DETECTION FAILED - Detailed Analysis:")
        logger.error(f"   - Image dimensions: {width}x{height}")
        logger.error(f"   - Image size: {len(image_bytes)} bytes")
        logger.error(f"   - Total detection attempts: {len(detection_attempts)}")
        
        # Group failures by error type
        no_face_errors = [a for a in detection_attempts if a["error_type"] == "no_face"]
        processing_errors = [a for a in detection_attempts if a["error_type"] == "processing_error"]
        invalid_face_errors = [a for a in detection_attempts if a["error_type"] == "invalid_face"]
        
        logger.error(f"   - No face detected: {len(no_face_errors)} attempts")
        logger.error(f"   - Processing errors: {len(processing_errors)} attempts")
        logger.error(f"   - Invalid faces (false positives): {len(invalid_face_errors)} attempts")
        
        # Log backend performance
        backend_stats = {}
        for attempt in detection_attempts:
            backend = attempt["backend"]
            if backend not in backend_stats:
                backend_stats[backend] = {"no_face": 0, "processing_error": 0, "invalid_face": 0}
            backend_stats[backend][attempt["error_type"]] += 1
        
        logger.error("   - Backend performance:")
        for backend, stats in backend_stats.items():
            logger.error(f"     * {backend}: {stats['no_face']} no-face, {stats['processing_error']} errors, {stats['invalid_face']} false-positives")
        
        # Log preprocessing attempts
        preprocessing_attempted = set([a["preprocessing"] for a in detection_attempts if a["preprocessing"] != "none"])
        if preprocessing_attempted:
            logger.error(f"   - Preprocessing methods attempted: {list(preprocessing_attempted)}")
        
        # Provide specific recommendations based on image characteristics
        recommendations = []
        if width < 200 or height < 200:
            recommendations.append("Image resolution too low (min 200x200 recommended)")
        if len(image_bytes) < 10000:  # Less than ~10KB
            recommendations.append("Image file size very small, might be over-compressed")
        
        # Check if many false positives were detected
        if len(invalid_face_errors) > len(no_face_errors):
            recommendations.append("Image contains objects that may be mistaken for faces (jewelry, patterns, etc.)")
            recommendations.append("Ensure image contains a clear, unobstructed human face")
        
        if recommendations:
            logger.error("   - Recommendations:")
            for rec in recommendations:
                logger.error(f"     * {rec}")
        
        # Sample some specific errors for debugging
        if detection_attempts:
            logger.error("   - Sample errors:")
            for i, attempt in enumerate(detection_attempts[:3]):  # Show first 3 errors
                logger.error(f"     * {attempt['backend']} ({attempt['preprocessing']}): {attempt['error']}")
        
        # If we had many invalid face detections, provide specific message
        if len(invalid_face_errors) >= 3:
            raise ValueError("no_valid_human_face_detected")
        else:
            raise ValueError("no_face_detected")

def check_employee_face(new_embedding, employee_df):
    """Enhanced employee face checking with better GPU utilization"""
    if employee_df.empty:
        logger.info("No employees registered for checking")
        return {"is_employee": False}

    all_employee_embeddings = np.array(employee_df["embedding"].tolist())
    logger.info(f"Checking against {len(all_employee_embeddings)} employee faces")
    
    # Use GPU for cosine similarity if available
    if CUDA_AVAILABLE:
        try:
            import torch
            device = torch.device("cuda:0")
            
            new_emb_tensor = torch.tensor([new_embedding], dtype=torch.float32).to(device)
            all_emb_tensor = torch.tensor(all_employee_embeddings, dtype=torch.float32).to(device)
            
            # Compute cosine similarity on GPU
            similarities = torch.nn.functional.cosine_similarity(
                new_emb_tensor.unsqueeze(1), 
                all_emb_tensor.unsqueeze(0), 
                dim=2
            ).cpu().numpy()[0]
            
            logger.info("Employee similarity computation performed on GPU")
        except Exception as e:
            logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            similarities = cosine_similarity([new_embedding], all_employee_embeddings)[0]
    else:
        similarities = cosine_similarity([new_embedding], all_employee_embeddings)[0]
    
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    logger.info(f"Best employee match score: {best_score:.4f} (threshold: {EMPLOYEE_SIMILARITY_THRESHOLD})")

    if best_score >= EMPLOYEE_SIMILARITY_THRESHOLD:
        logger.warning(f"Employee match found: {employee_df.iloc[best_idx]['employee_name']} (score: {best_score:.4f})")
        return {
            "is_employee": True,
            "employee_id": employee_df.iloc[best_idx]["employee_id"],
            "employee_name": employee_df.iloc[best_idx]["employee_name"],
            "score": float(best_score),
        }
    
    logger.info("No employee match found")
    return {"is_employee": False}

def find_duplicate(new_embedding, df):
    """Enhanced duplicate detection with better GPU utilization"""
    if df.empty:
        logger.info("No users registered for duplicate checking")
        return {"duplicate": False}

    all_embeddings = np.array(df["embedding"].tolist())
    logger.info(f"Checking for duplicates against {len(all_embeddings)} registered users")
    
    # Use GPU for cosine similarity if available
    if CUDA_AVAILABLE:
        try:
            import torch
            device = torch.device("cuda:0")
            
            new_emb_tensor = torch.tensor([new_embedding], dtype=torch.float32).to(device)
            all_emb_tensor = torch.tensor(all_embeddings, dtype=torch.float32).to(device)
            
            # Compute cosine similarity on GPU
            similarities = torch.nn.functional.cosine_similarity(
                new_emb_tensor.unsqueeze(1), 
                all_emb_tensor.unsqueeze(0), 
                dim=2
            ).cpu().numpy()[0]
            
            logger.info("User similarity computation performed on GPU")
        except Exception as e:
            logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            similarities = cosine_similarity([new_embedding], all_embeddings)[0]
    else:
        similarities = cosine_similarity([new_embedding], all_embeddings)[0]
    
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    logger.info(f"Best user match score: {best_score:.4f} (threshold: {SIMILARITY_THRESHOLD})")

    if best_score >= SIMILARITY_THRESHOLD:
        logger.info(f"Duplicate found: {df.iloc[best_idx]['user_id']} (score: {best_score:.4f})")
        return {
            "duplicate": True,
            "matched_user_id": df.iloc[best_idx]["user_id"],
            "score": float(best_score),
        }
    
    logger.info("No duplicate found")
    return {"duplicate": False}

# ---------------------- Health Endpoint ----------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system information"""
    try:
        # Test if we can load embeddings
        user_df = load_embeddings()
        employee_df = load_employee_embeddings()
        total_user_embeddings = len(user_df)
        total_employee_embeddings = len(employee_df)
        
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
                "total_registered_users": total_user_embeddings,
                "total_registered_employees": total_employee_embeddings,
                "user_embedding_file_exists": os.path.exists(EMBEDDING_FILE),
                "employee_embedding_file_exists": os.path.exists(EMPLOYEE_EMBEDDING_FILE)
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
        "user_similarity_threshold": SIMILARITY_THRESHOLD,
        "employee_similarity_threshold": EMPLOYEE_SIMILARITY_THRESHOLD,
        "cuda_support": CUDA_AVAILABLE,
        "cuda_devices": CUDA_DEVICE_COUNT,
        "gpu_name": CUDA_DEVICE_NAME,
        "processing_device": "GPU" if CUDA_AVAILABLE else "CPU",
        "available_detection_backends": [
            "retinaface", "mtcnn", "opencv", "ssd", 
            "dlib", "mediapipe", "yolov8", "fastmtcnn"
        ],
        "preprocessing_methods": [
            "histogram_equalization", "clahe", "brightness_contrast",
            "gamma_correction", "blur_sharpen"
        ]
    }

# ---------------------- Employee Registration Endpoints ----------------------
@app.post("/register-employee")
async def register_employee(request: EmployeeRegistrationRequest):
    """Register an employee face to the blocked list using image URL"""
    start_time = datetime.now()
    
    # Step 1: Download image
    try:
        logger.info(f"Downloading employee image from URL for employee: {request.employee_id}")
        response = requests.get(request.image_url, timeout=30)
        if response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={"status": "failed_to_download_image_from_url"}
            )
        image_bytes = response.content
    except Exception as e:
        logger.error(f"Failed to download employee image: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "failed_to_download_image_from_url"}
        )

    # Step 2: Generate embedding with enhanced detection
    try:
        logger.info("Generating employee face embedding...")
        new_embedding = get_embedding_from_bytes(image_bytes)
    except ValueError as e:
        error_status = str(e)
        if error_status == "no_face_detected":
            return JSONResponse(
                status_code=400,
                content={"status": "no_face_detected"}
            )
        elif error_status == "no_valid_human_face_detected":
            return JSONResponse(
                status_code=400,
                content={
                    "status": "no_valid_human_face_detected", 
                    "message": "Detected objects were not valid human faces (possibly jewelry, patterns, or other objects)"
                }
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

    # Step 3: Save employee embedding
    logger.info(f"Registering employee: {request.employee_id}")
    employee_df = load_employee_embeddings()
    
    # Check if employee already exists
    if not employee_df.empty and request.employee_id in employee_df["employee_id"].values:
        return JSONResponse(
            status_code=400,
            content={"status": "employee_already_registered", "employee_id": request.employee_id}
        )
    
    new_employee_row = pd.DataFrame([{
        "employee_id": request.employee_id, 
        "employee_name": request.employee_name,
        "embedding": new_embedding
    }])
    employee_df = pd.concat([employee_df, new_employee_row], ignore_index=True)
    save_employee_embeddings(employee_df)

    processing_time = (datetime.now() - start_time).total_seconds()

    return JSONResponse(content={
        "status": "employee_registered_successfully",
        "employee_id": request.employee_id,
        "employee_name": request.employee_name,
        "processing_time_seconds": processing_time,
        "total_employees": len(employee_df)
    })

@app.post("/register-employee-file")
async def register_employee_file(
    employee_id: str,
    employee_name: str,
    file: UploadFile = File(...)
):
    """Register an employee face to the blocked list using file upload"""
    start_time = datetime.now()
    
    # Step 1: Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ''
    
    if file_extension not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid_file_type", 
                "message": "Only image files are allowed (jpg, jpeg, png, bmp, tiff, webp)",
                "received_extension": file_extension
            }
        )
    
    # Step 2: Read file content
    try:
        logger.info(f"Processing uploaded file for employee: {employee_id}")
        image_bytes = await file.read()
        
        # Validate file size (10MB limit)
        if len(image_bytes) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"status": "file_too_large", "message": "File size must be less than 10MB"}
            )
            
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "failed_to_read_file", "error": str(e)}
        )

    # Step 3: Generate embedding with enhanced detection
    try:
        logger.info("Generating employee face embedding from uploaded file...")
        new_embedding = get_embedding_from_bytes(image_bytes)
    except ValueError as e:
        error_status = str(e)
        if error_status == "no_face_detected":
            return JSONResponse(
                status_code=400,
                content={"status": "no_face_detected", "message": "No face detected in the uploaded image"}
            )
        elif error_status == "no_valid_human_face_detected":
            return JSONResponse(
                status_code=400,
                content={
                    "status": "no_valid_human_face_detected", 
                    "message": "Detected objects were not valid human faces (possibly jewelry, patterns, or other objects)"
                }
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

    # Step 4: Save employee embedding
    logger.info(f"Registering employee from file: {employee_id}")
    employee_df = load_employee_embeddings()
    
    # Check if employee already exists
    if not employee_df.empty and employee_id in employee_df["employee_id"].values:
        return JSONResponse(
            status_code=400,
            content={"status": "employee_already_registered", "employee_id": employee_id}
        )
    
    new_employee_row = pd.DataFrame([{
        "employee_id": employee_id, 
        "employee_name": employee_name,
        "embedding": new_embedding
    }])
    employee_df = pd.concat([employee_df, new_employee_row], ignore_index=True)
    save_employee_embeddings(employee_df)

    processing_time = (datetime.now() - start_time).total_seconds()

    return JSONResponse(content={
        "status": "employee_registered_successfully",
        "employee_id": employee_id,
        "employee_name": employee_name,
        "filename": file.filename,
        "file_size_mb": round(len(image_bytes) / (1024 * 1024), 2),
        "processing_time_seconds": processing_time,
        "total_employees": len(employee_df)
    })

# ---------------------- Main Endpoint (Enhanced) ----------------------
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

    # Step 2: Generate embedding with enhanced detection
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

    # Step 3: Check if face belongs to an employee
    logger.info("Checking if face belongs to an employee...")
    employee_df = load_employee_embeddings()
    employee_check = check_employee_face(new_embedding, employee_df)
    
    if employee_check.get("is_employee"):
        logger.warning(f"Employee face detected - registration declined for user: {request.user_id}")
        return JSONResponse(
            status_code=403,
            content={
                "status": "employee_face_detected",
                "message": "Employee faces are not allowed for user registration",
                "employee_id": employee_check["employee_id"],
                "employee_name": employee_check["employee_name"],
                "similarity_score": employee_check["score"]
            }
        )

    # Step 4: Check for duplicates among users
    logger.info("Checking for duplicate faces among users...")
    user_df = load_embeddings()
    result = find_duplicate(new_embedding, user_df)

    processing_time = (datetime.now() - start_time).total_seconds()

    if result.get("duplicate"):
        logger.info(f"Duplicate face found for user: {request.user_id}")
        return JSONResponse(content={
            "status": "duplicate_image_found",
            "matched_user_id": result["matched_user_id"],
            "score": result["score"],
            "processing_time_seconds": processing_time
        })

    # Step 5: Save new user embedding
    logger.info(f"Registering new face for user: {request.user_id}")
    new_row = pd.DataFrame([{"user_id": request.user_id, "embedding": new_embedding}])
    user_df = pd.concat([user_df, new_row], ignore_index=True)
    save_embeddings(user_df)

    return JSONResponse(content={
        "status": "new_face_registered",
        "user_id": request.user_id,
        "processing_time_seconds": processing_time,
        "total_users": len(user_df)
    })

# ---------------------- File Upload Endpoint ----------------------
@app.post("/check-face-file")
async def check_face_file(
    user_id: str,
    file: UploadFile = File(...)
):
    """Check face using file upload instead of URL"""
    start_time = datetime.now()
    
    # Step 1: Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ''
    
    if file_extension not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid_file_type", 
                "message": "Only image files are allowed (jpg, jpeg, png, bmp, tiff, webp)",
                "received_extension": file_extension
            }
        )
    
    # Step 2: Read file content
    try:
        logger.info(f"Processing uploaded file for user: {user_id}")
        image_bytes = await file.read()
        
        # Validate file size (10MB limit)
        if len(image_bytes) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"status": "file_too_large", "message": "File size must be less than 10MB"}
            )
            
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "failed_to_read_file", "error": str(e)}
        )

    # Step 3: Generate embedding with enhanced detection
    try:
        logger.info("Generating face embedding from uploaded file...")
        new_embedding = get_embedding_from_bytes(image_bytes)
    except ValueError as e:
        error_status = str(e)
        if error_status == "no_face_detected":
            return JSONResponse(
                status_code=400,
                content={"status": "no_face_detected", "message": "No face detected in the uploaded image"}
            )
        elif error_status == "no_valid_human_face_detected":
            return JSONResponse(
                status_code=400,
                content={
                    "status": "no_valid_human_face_detected", 
                    "message": "Detected objects were not valid human faces (possibly jewelry, patterns, or other objects)"
                }
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

    # Step 4: Check if face belongs to an employee
    logger.info("Checking if face belongs to an employee...")
    employee_df = load_employee_embeddings()
    employee_check = check_employee_face(new_embedding, employee_df)
    
    if employee_check.get("is_employee"):
        logger.warning(f"Employee face detected - registration declined for user: {user_id}")
        return JSONResponse(
            status_code=403,
            content={
                "status": "employee_face_detected",
                "message": "Employee faces are not allowed for user registration",
                "employee_id": employee_check["employee_id"],
                "employee_name": employee_check["employee_name"],
                "similarity_score": employee_check["score"]
            }
        )

    # Step 5: Check for duplicates among users
    logger.info("Checking for duplicate faces among users...")
    user_df = load_embeddings()
    result = find_duplicate(new_embedding, user_df)

    processing_time = (datetime.now() - start_time).total_seconds()

    if result.get("duplicate"):
        logger.info(f"Duplicate face found for user: {user_id}")
        return JSONResponse(content={
            "status": "duplicate_image_found",
            "matched_user_id": result["matched_user_id"],
            "score": result["score"],
            "processing_time_seconds": processing_time,
            "filename": file.filename
        })

    # Step 6: Save new user embedding
    logger.info(f"Registering new face for user: {user_id}")
    new_row = pd.DataFrame([{"user_id": user_id, "embedding": new_embedding}])
    user_df = pd.concat([user_df, new_row], ignore_index=True)
    save_embeddings(user_df)

    return JSONResponse(content={
        "status": "new_face_registered",
        "user_id": user_id,
        "filename": file.filename,
        "file_size_mb": round(len(image_bytes) / (1024 * 1024), 2),
        "processing_time_seconds": processing_time,
        "total_users": len(user_df)
    })

# ---------------------- Employee Management Endpoints ----------------------
@app.get("/employees")
async def list_employees():
    """List all registered employees"""
    employee_df = load_employee_embeddings()
    if employee_df.empty:
        return {"employees": [], "total_count": 0}
    
    employees = []
    for _, row in employee_df.iterrows():
        employees.append({
            "employee_id": row["employee_id"],
            "employee_name": row["employee_name"]
        })
    
    return {"employees": employees, "total_count": len(employees)}

@app.delete("/employee/{employee_id}")
async def remove_employee(employee_id: str):
    """Remove an employee from the blocked list"""
    employee_df = load_employee_embeddings()
    
    if employee_df.empty or employee_id not in employee_df["employee_id"].values:
        return JSONResponse(
            status_code=404,
            content={"status": "employee_not_found", "employee_id": employee_id}
        )
    
    # Remove the employee
    updated_df = employee_df[employee_df["employee_id"] != employee_id]
    save_employee_embeddings(updated_df)
    
    return JSONResponse(content={
        "status": "employee_removed_successfully",
        "employee_id": employee_id,
        "remaining_employees": len(updated_df)
    })

# ---------------------- User Management Endpoints ----------------------
@app.get("/users")
async def list_users():
    """List all registered users"""
    user_df = load_embeddings()
    if user_df.empty:
        return {"users": [], "total_count": 0}
    
    users = []
    for _, row in user_df.iterrows():
        users.append({"user_id": row["user_id"]})
    
    return {"users": users, "total_count": len(users)}

@app.delete("/user/{user_id}")
async def remove_user(user_id: str):
    """Remove a user from the registered users"""
    user_df = load_embeddings()
    
    if user_df.empty or user_id not in user_df["user_id"].values:
        return JSONResponse(
            status_code=404,
            content={"status": "user_not_found", "user_id": user_id}
        )
    
    # Remove the user
    updated_df = user_df[user_df["user_id"] != user_id]
    save_embeddings(updated_df)
    
    return JSONResponse(content={
        "status": "user_removed_successfully",
        "user_id": user_id,
        "remaining_users": len(updated_df)
    })

# ---------------------- Startup Event ----------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Enhanced Face Recognition API starting up...")
    logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        logger.info(f"CUDA Devices: {CUDA_DEVICE_COUNT}")
        logger.info(f"GPU: {CUDA_DEVICE_NAME}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Primary Detector: {DETECTOR}")
    logger.info(f"User Similarity Threshold: {SIMILARITY_THRESHOLD}")
    logger.info(f"Employee Similarity Threshold: {EMPLOYEE_SIMILARITY_THRESHOLD}")
    logger.info("Detection Backends: retinaface, mtcnn, opencv, ssd, dlib, mediapipe, yolov8, fastmtcnn")
    logger.info("Preprocessing Methods: histogram_equalization, clahe, brightness_contrast, gamma_correction, blur_sharpen")
    
    # Load existing embeddings on startup
    user_df = load_embeddings()
    employee_df = load_employee_embeddings()
    logger.info(f"Loaded {len(user_df)} user embeddings and {len(employee_df)} employee embeddings")

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