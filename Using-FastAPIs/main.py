
import logging
from datetime import datetime
import requests
import pandas as pd

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

import config
import face_processor
import embedding_manager as db
from face_processor import BlurryImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Modular Face Recognition API",
    description="An accurate API with a standardized JSON response structure.",
    version="3.0.0" # Version bump for new response structure
)

# --- Pydantic Models for URL requests (no changes here) ---
class ImageURLRequest(BaseModel):
    user_id: str
    image_url: HttpUrl

class EmployeeURLRequest(BaseModel):
    employee_id: str
    employee_name: str
    image_url: HttpUrl

# --- Helper to download image (no changes here) ---
def download_image(url: HttpUrl) -> bytes:
    try:
        response = requests.get(str(url), timeout=15)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"failed_to_download_image")
        raise HTTPException(status_code=400, detail="failed_to_download_image")

# --- Main Endpoints (User) ---
@app.post("/check-face-file")
async def check_face_file(user_id: str = Form(...), file: UploadFile = File(...)):
    """Checks face from file, blocks employees, finds duplicates, and registers new users."""
    image_bytes = await file.read()
    return await process_face_check(user_id, image_bytes)

@app.post("/check-face-url")
async def check_face_url(request: ImageURLRequest):
    """Checks face from URL, blocks employees, finds duplicates, and registers new users."""
    image_bytes = download_image(request.image_url)
    return await process_face_check(request.user_id, image_bytes)

async def process_face_check(user_id: str, image_bytes: bytes):
    """Core logic for face checking and registration with the new response structure."""
    start_time = datetime.now()
    try:
        embedding = face_processor.get_embedding(image_bytes)
    
    except BlurryImageError as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": False,
                "user_id": user_id,
                "data": {"blur_score": round(e.score, 2)},
                "message": "blur_image_detected"
            }
        )
    except ValueError as e:
        # Distinguish between "no face" and other processing errors for a better message
        error_message = "no_face_detected" if "No_face_detected" in str(e) else "image_processing_error"
        return JSONResponse(
            status_code=400,
            content={
                "status": False,
                "user_id": user_id,
                "data": {"details": ""},
                "message": error_message
            }
        )

    # Block employee faces
    if (employee_match := db.check_is_employee(embedding)):
        return JSONResponse(
            status_code=403, # Forbidden
            content={
                "status": False,
                # "user_id": user_id
                "data": employee_match,
                "message": "employee_face_detected"
            }
        )

    # Find duplicate users
    if (duplicate := db.find_duplicate_user(embedding)):
        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                # "user_id": user_id,
                "data": duplicate,
                "message": "duplicate_face_found"
            }
        )
    
    # Register new user
    user_df = db.load_user_embeddings()
    new_row = pd.DataFrame([{"user_id": user_id, "embedding": embedding}])
    user_df = pd.concat([user_df, new_row], ignore_index=True)
    db.save_user_embeddings(user_df)

    processing_time = (datetime.now() - start_time).total_seconds()
    return JSONResponse(
        status_code=200,
        content={
            "status": True,
            "user_id": user_id,
            "data": {"processing_time_seconds": round(processing_time, 2)},
            "message": "new_face_registered"
        }
    )

# --- Employee Registration ---
@app.post("/register-employee")
async def register_employee(request: EmployeeURLRequest):
    """Registers an employee face from a URL."""
    image_bytes = download_image(request.image_url)
    return await process_employee_registration(request.employee_id, request.employee_name, image_bytes)

@app.post("/register-employee-file")
async def register_employee_file(employee_id: str = Form(...), employee_name: str = Form(...), file: UploadFile = File(...)):
    """Registers an employee face from a file."""
    image_bytes = await file.read()
    return await process_employee_registration(employee_id, employee_name, image_bytes)

async def process_employee_registration(employee_id: str, employee_name: str, image_bytes: bytes):
    """Core logic for employee registration with the new response structure."""
    try:
        embedding = face_processor.get_embedding(image_bytes)
    
    except BlurryImageError as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": False,
                "employee_id": employee_id,
                "data": {"blur_score": round(e.score, 2)},
                "message": "blur_image_detected"
            }
        )
    except ValueError as e:
        error_message = "no_face_detected" if "No face detected" in str(e) else "image_processing_error"
        return JSONResponse(
            status_code=400,
            content={
                "status": False,
                "employee_id": employee_id,
                "data": {"details": ""},
                "message": error_message
            }
        )
    
    employee_df = db.load_employee_embeddings()
    if not employee_df.empty and employee_id in employee_df["employee_id"].values:
        return JSONResponse(
            status_code=409, # Conflict
            content={
                "status": False,
                "employee_id": employee_id,
                "data": {},
                "message": "employee_already_exists"
            }
        )

    new_row = pd.DataFrame([{"employee_id": employee_id, "employee_name": employee_name, "embedding": embedding}])
    employee_df = pd.concat([employee_df, new_row], ignore_index=True)
    db.save_employee_embeddings(employee_df)
    
    return JSONResponse(
        status_code=200,
        content={
            "status": True,
            "employee_id": employee_id,
            "data": {"employee_name": employee_name},
            "message": "employee_registered_successfully"
        }
    )

# --- Management Endpoints ---
@app.get("/employees")
async def list_employees():
    """Lists all registered employees."""
    df = db.load_employee_embeddings()
    employee_list = []
    if not df.empty:
        employee_list = df[["employee_id", "employee_name"]].to_dict(orient="records")
    
    return {
        "status": True,
        "data": {"employees": employee_list},
        "message": "employees_retrieved_successfully"
    }

@app.delete("/employee/{employee_id}")
async def remove_employee(employee_id: str):
    """Removes an employee by their ID."""
    df = db.load_employee_embeddings()
    if employee_id not in df["employee_id"].values:
        return JSONResponse(
            status_code=404,
            content={
                "status": False,
                "employee_id": employee_id,
                "data": {},
                "message": "employee_not_found"
            }
        )
    
    updated_df = df[df["employee_id"] != employee_id]
    db.save_employee_embeddings(updated_df)
    return {
        "status": True,
        "employee_id": employee_id,
        "data": {},
        "message": "employee_removed_successfully"
    }

@app.delete("/user/{user_id}")
async def remove_user(user_id: str):
    """Removes a user by their ID."""
    df = db.load_user_embeddings()
    if user_id not in df["user_id"].values:
        return JSONResponse(
            status_code=404,
            content={
                "status": False,
                "user_id": user_id,
                "data": {},
                "message": "user_not_found"
            }
        )

    updated_df = df[df["user_id"] != user_id]
    db.save_user_embeddings(updated_df)
    return {
        "status": True,
        "user_id": user_id,
        "data": {},
        "message": "user_removed_successfully"
    }

# --- Startup Event (no changes here) ---
@app.on_event("startup")
async def startup_event():
    logger.info("Face Recognition API starting up...")
    logger.info(f"Model: {config.MODEL_NAME}, Primary Detector: {config.PRIMARY_DETECTOR}")
    logger.info(f"User Similarity: {config.USER_SIMILARITY_THRESHOLD}, Blur Threshold: {config.BLUR_THRESHOLD}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8100, 
        reload=True,
        log_level="info"
    )