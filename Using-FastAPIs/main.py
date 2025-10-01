#main.py
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
    version="3.1.0" # Version bump for name support
)

# --- Pydantic Models for URL requests ---
class ImageURLRequest(BaseModel):
    user_id: str
    name: str
    image_url: HttpUrl

class EmployeeURLRequest(BaseModel):
    employee_id: str
    employee_name: str
    image_url: HttpUrl

# --- Helper to download image ---
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
async def check_face_file(user_id: str = Form(...), name: str = Form(...), file: UploadFile = File(...)):
    """Checks face from file, blocks employees, finds duplicates, and registers new users."""
    image_bytes = await file.read()
    return await process_face_check(user_id, name, image_bytes)

@app.post("/check-face-url")
async def check_face_url(request: ImageURLRequest):
    """Checks face from URL, blocks employees, finds duplicates, and registers new users."""
    image_bytes = download_image(request.image_url)
    return await process_face_check(request.user_id, request.name, image_bytes)

# Updated sections for main.py

from face_processor import BlurryImageError, MultipleFacesError

async def process_face_check(user_id: str, name: str, image_bytes: bytes):
    """Core logic for face checking and registration with multiple face handling."""
    start_time = datetime.now()
    try:
        embedding = face_processor.get_embedding(image_bytes)
    
        # Block employee faces
        if (employee_match := db.check_is_employee(embedding)):
            return JSONResponse(
                status_code=403, # Forbidden
                content={
                    "success": True,
                    "data": {
                        "matched_id": employee_match.get("employee_id", ""),
                        "name": employee_match.get("employee_name", ""),
                    },
                    "message": "employee_face_detected"
                }
            )

        # Find duplicate users
        if (duplicate := db.find_duplicate_user(embedding)):
            # Load user data to get the name of the duplicate user
            user_df = db.load_user_embeddings()
            duplicate_name = ""
            if not user_df.empty:
                duplicate_row = user_df[user_df["user_id"] == duplicate.get("matched_user_id", "")]
                if not duplicate_row.empty:
                    duplicate_name = duplicate_row.iloc[0].get("name", "")
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": {
                        "matched_id": duplicate.get("matched_user_id", ""),
                        "name": duplicate_name,
                        "similarity_score": round(duplicate.get("score", 0), 3)
                    },
                    "message": "duplicate_face_found"
                }
            )
        
        # Register new user
        user_df = db.load_user_embeddings()
        new_row = pd.DataFrame([{"user_id": user_id, "name": name, "embedding": embedding}])
        user_df = pd.concat([user_df, new_row], ignore_index=True)
        db.save_user_embeddings(user_df)

        processing_time = (datetime.now() - start_time).total_seconds()
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "user_id": user_id,
                    "name": name,
                },
                "message": "new_face_registered"
            }
        )
    
    except MultipleFacesError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {
                    "face_count": e.face_count,
                    # "faces_detected": e.faces_info,
                    # "total_faces": len(e.faces_info) if hasattr(e, 'faces_info') else e.face_count
                },
                "message": "multiple_faces_detected"
            }
        )
    except BlurryImageError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {
                    "blur_score": round(e.score, 2),
                    "threshold": config.BLUR_THRESHOLD
                },
                "message": "blur_image_detected"
            }
        )
    except ValueError as e:
        # Distinguish between different error types
        if "No face detected" in str(e):
            error_message = "no_face_detected"
        elif "No prominent face detected" in str(e):
            error_message = "no_prominent_face_detected"
        else:
            error_message = "image_processing_error"
            
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {},
                "message": error_message
            }
        )

async def process_employee_registration(employee_id: str, employee_name: str, image_bytes: bytes):
    """Core logic for employee registration with multiple face handling."""
    try:
        embedding = face_processor.get_embedding(image_bytes)
    
    except MultipleFacesError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {
                    # "face_count": e.face_count,
                    # "faces_detected": e.faces_info,
                    "total_faces": len(e.faces_info) if hasattr(e, 'faces_info') else e.face_count
                },
                "message": "multiple_faces_detected"
            }
        )
    except BlurryImageError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {
                    "blur_score": round(e.score, 2),
                    "threshold": config.BLUR_THRESHOLD
                },
                "message": "blur_image_detected"
            }
        )
    except ValueError as e:
        if "No face detected" in str(e):
            error_message = "no_face_detected"
        elif "No prominent face detected" in str(e):
            error_message = "no_prominent_face_detected"
        else:
            error_message = "image_processing_error"
            
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {},
                "message": error_message
            }
        )
    
    employee_df = db.load_employee_embeddings()
    if not employee_df.empty and employee_id in employee_df["employee_id"].values:
        return JSONResponse(
            status_code=409, # Conflict
            content={
                "success": False,
                "data": {
                    "employee_id": employee_id,
                    "employee_name": employee_name
                },
                "message": "employee_already_exists"
            }
        )

    new_row = pd.DataFrame([{"employee_id": employee_id, "employee_name": employee_name, "embedding": embedding}])
    employee_df = pd.concat([employee_df, new_row], ignore_index=True)
    db.save_employee_embeddings(employee_df)
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "data": {
                "employee_id": employee_id,
                "employee_name": employee_name
            },
            "message": "employee_registered_successfully"
        }
    )

@app.get("/employees")
async def list_employees():
    """Lists all registered employees."""
    df = db.load_employee_embeddings()
    employee_list = []
    if not df.empty:
        employee_list = df[["employee_id", "employee_name"]].to_dict(orient="records")
    
    return {
        "success": True,
        "data": {
            "employees": employee_list,
        },
        "message": "employees_retrieved_successfully"
    }

@app.get("/users")
async def list_users():
    """Lists all registered users."""
    df = db.load_user_embeddings()
    user_list = []
    if not df.empty:
        user_list = df[["user_id", "name"]].to_dict(orient="records")
    
    return {
        "success": True,
        "data": {
            "users": user_list,
        },
        "message": "users_retrieved_successfully"
    }

@app.delete("/employee/{employee_id}")
async def remove_employee(employee_id: str):
    """Removes an employee by their ID."""
    df = db.load_employee_embeddings()
    if employee_id not in df["employee_id"].values:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "data": {
                    "employee_id": employee_id
                },
                "message": "employee_not_found"
            }
        )
    
    # Get employee name before deletion
    employee_name = df[df["employee_id"] == employee_id].iloc[0]["employee_name"]
    
    updated_df = df[df["employee_id"] != employee_id]
    db.save_employee_embeddings(updated_df)
    return {
        "success": True,
        "data": {
            "employee_id": employee_id,
            "employee_name": employee_name
        },
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
                "success": False,
                "data": {
                    "user_id": user_id
                },
                "message": "user_not_found"
            }
        )

    # Get user name before deletion
    name = df[df["user_id"] == user_id].iloc[0]["name"]
    
    updated_df = df[df["user_id"] != user_id]
    db.save_user_embeddings(updated_df)
    return {
        "success": True,
        "data": {
            "user_id": user_id,
            "name": name
        },
        "message": "user_removed_successfully"
    }

# --- Startup Event ---
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
        port=8102, 
        reload=True,
        log_level="info"
    )