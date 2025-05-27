import cv2
from deepface import DeepFace
import os
from PIL import Image

# Original path
original_path = "/Users/hqpl/Downloads/archive/Image_720.jpg"

# Clean and save
img = Image.open(original_path).convert("RGB")
clean_path = "/Users/hqpl/Downloads/archive/lakshya_clean.jpg"
img.save(clean_path, format="JPEG")
print("✅ Reference image saved at:", clean_path)


# Cleaned reference image path
reference_img_path = "/Users/hqpl/Downloads/archive/lakshya_clean.jpg"

# Check if reference image exists
if not os.path.exists(reference_img_path):
    print("❌ Reference image not found!")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 's' to capture an image and verify face")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Live Feed - Press 's' to capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Save the captured frame
        live_img_path = "captured_face.jpg"
        cv2.imwrite(live_img_path, frame)
        print("📸 Image Captured. Verifying...")

        try:
            result = DeepFace.verify(
                img1_path=reference_img_path,
                img2_path=live_img_path,
                detector_backend='retinaface'  # Change to 'mtcnn' or 'mediapipe' if needed
            )
            print("🔍 Result:", result)
            if result["verified"]:
                print("✅ Face Matched!")
            else:
                print("❌ Face Not Matched!")

        except Exception as e:
            print("⚠️ Verification failed:", str(e))

    elif key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()


