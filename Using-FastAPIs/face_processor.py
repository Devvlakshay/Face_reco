# face_processor.py
import logging
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from deepface import DeepFace

import config

logger = logging.getLogger(__name__)

# Custom Exception for Blurry Images
class BlurryImageError(ValueError):
    """Custom exception raised when an image's blurriness is below the threshold."""
    def __init__(self, message, score):
        super().__init__(message)
        self.score = score

def is_image_blurry(image_array: np.ndarray):
    """
    Detects if an image is too blurry by calculating the variance of the Laplacian.
    This should be run on a CROPPED FACE, not the whole image.
    """
    # Convert to grayscale for blur detection
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # Calculate the Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    logger.info(f"Image blur score (on face crop): {laplacian_var:.2f} (Threshold: {config.BLUR_THRESHOLD})")
    
    if laplacian_var < config.BLUR_THRESHOLD:
        return True, laplacian_var
    return False, laplacian_var

def resize_image(image: Image.Image, max_size=1024) -> Image.Image:
    """Resizes an image to a maximum dimension while preserving aspect ratio."""
    if image.width > max_size or image.height > max_size:
        logger.info(f"Image is large ({image.width}x{image.height}), resizing to max {max_size}px.")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def get_embedding(image_bytes: bytes):
    """
    Generates a facial embedding with a more robust validation pipeline:
    1. Resize image.
    2. Detect a face using a sequence of detectors.
    3. If face is found, CROP the face region.
    4. Run blur detection ONLY on the cropped face.
    5. Generate embedding.
    """
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ValueError("Invalid or corrupted image file.")

    # Step 1: Resize image for normalization
    pil_image = resize_image(pil_image)
    img_array = np.array(pil_image)
    
    # --- MODIFIED LOGIC: Step 2 is now Face Detection ---
    detection_backends = [config.PRIMARY_DETECTOR, config.SECONDARY_DETECTOR, config.TERTIARY_DETECTOR]
    detected_face = None

    for backend in detection_backends:
        try:
            logger.info(f"Attempting face detection with '{backend}'...")
            faces = DeepFace.extract_faces(
                img_path=img_array,
                detector_backend=backend,
                enforce_detection=True
            )
            
            best_face = max(faces, key=lambda x: x.get('confidence', 0.0))
            confidence = best_face.get('confidence', 0.0)
            
            if backend == 'dlib' and len(faces) > 0:
                logger.info(f"Face detected via '{backend}'. Accepting without confidence score.")
                detected_face = best_face
                break

            logger.info(f"Face detected via '{backend}' with confidence: {confidence:.4f}")
            
            if confidence >= config.FACE_CONFIDENCE_THRESHOLD:
                detected_face = best_face
                break
            else:
                logger.warning(f"Face confidence from '{backend}' ({confidence:.2f}) is below threshold.")

        except ValueError:
            logger.warning(f"No face detected using '{backend}'.")
            continue
    
    if not detected_face:
        raise ValueError("No face detected with sufficient confidence across all detectors.")
        
    # --- NEW LOGIC: Step 3 is Blur Detection on the CROPPED FACE ---
    facial_area = detected_face['facial_area']
    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
    face_crop = img_array[y:y+h, x:x+w]
    
    is_blurry, score = is_image_blurry(face_crop)
    if is_blurry:
        message = f"Image is too blurry to process (score: {score:.2f}, threshold: {config.BLUR_THRESHOLD})."
        raise BlurryImageError(message=message, score=score)

    # --- Step 4: Generate the embedding (no changes here) ---
    try:
        embedding_obj = DeepFace.represent(
            img_path=img_array,
            model_name=config.MODEL_NAME,
            detector_backend=config.PRIMARY_DETECTOR, # Backend here doesn't matter much as we enforce_detection=False
            enforce_detection=False
        )
        return embedding_obj[0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding generation failed after successful detection: {e}")
        raise ValueError("Face was detected, but embedding could not be generated.")