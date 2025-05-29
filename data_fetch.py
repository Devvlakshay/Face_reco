import os
import boto3
from PIL import Image
from io import BytesIO
import pickle
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

from dotenv import load_dotenv
load_dotenv()

# Environment Variables (or manually set here)
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")

# Output directories
IMAGE_DIR = "downloaded_images"
PKL_FILE = "face_embeddings.pkl"

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Boto3 client for Cloudflare R2
session = boto3.session.Session()
client = session.client(
    service_name='s3',
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    endpoint_url=R2_ENDPOINT,
    region_name='auto'
)

def download_images():
    print("[1/2] Downloading images from R2...")
    response = client.list_objects_v2(Bucket=R2_BUCKET)
    image_files = []

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.lower().endswith(('.jpg', '.jpeg', '.png')):
            local_path = os.path.join(IMAGE_DIR, os.path.basename(key))
            if not os.path.exists(local_path):
                try:
                    s3_obj = client.get_object(Bucket=R2_BUCKET, Key=key)
                    image_data = s3_obj['Body'].read()
                    with open(local_path, "wb") as f:
                        f.write(image_data)
                    print(f"Downloaded: {key}")
                    image_files.append(local_path)
                except Exception as e:
                    print(f"Failed to download {key}: {e}")
            else:
                print(f"Already exists: {local_path}")
                image_files.append(local_path)
    return image_files


if __name__ == "__main__":
    img_paths = download_images()

