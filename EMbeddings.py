import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# -----------------------------
# Config
img_folder = "/Users/hqpl/Downloads/archive/train/n000002"  # folder containing known user images
input_img_path = "/Users/hqpl/Downloads/archive/train/n000002/0001_01.jpg"  # image to verify for duplicates
embedding_file = "user_embeddings.pkl"
model_name = "Facenet"
detector = "retinaface"
similarity_threshold = 0.8  # Set a realistic threshold
# -----------------------------

# Extract and store embeddings from known user images
def extract_and_store_embeddings():
    embeddings = []
    user_ids = []

    for file in os.listdir(img_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(img_folder, file)
            try:
                embedding = DeepFace.represent(
                    img_path=path,
                    model_name=model_name,
                    detector_backend=detector
                )[0]["embedding"]
                embeddings.append(embedding)
                user_ids.append(file)
            except Exception as e:
                print(f"❌ Error with {file}: {e}")

    data = {
        "user_id": user_ids,
        "embedding": embeddings
    }

    with open(embedding_file, "wb") as f:
        pickle.dump(data, f)

    print("✅ Embeddings stored successfully.")

# Load embeddings from disk
def load_embeddings():
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    return pd.DataFrame(data)

# Compare input image embedding with stored embeddings
def check_duplicate(new_user_img_path, df, threshold=similarity_threshold):
    print("📸 Checking input image for duplicates...")

    try:
        new_embedding = DeepFace.represent(
            img_path=new_user_img_path,
            model_name=model_name,
            detector_backend=detector
        )[0]["embedding"]
    except Exception as e:
        print("❌ Failed to get embedding:", e)
        return

    all_embeddings = np.array(df["embedding"].tolist())
    similarities = cosine_similarity([new_embedding], all_embeddings)[0]

    # Get top 2 most similar images
        # Get top 2 most similar images
    top_indices = np.argsort(similarities)[::-1][:2]
    print("\n🔝 Top 2 most similar images:")
    found = False
    for rank, idx in enumerate(top_indices, 1):
        user_id = df.iloc[idx]['user_id']
        score = similarities[idx]
        # Construct the full path
        img_path = os.path.join(img_folder, user_id)
        print(f"{rank}. {img_path} - Cosine Similarity: {score:.4f}")
        if score >= threshold:
            print(f"⚠️ Duplicate found: {img_path} (Score: {score:.4f})")
            found = True

    found = False
    for idx, score in enumerate(similarities):
        user_id = df.iloc[idx]['user_id']
        print(f"{user_id} - Cosine Similarity: {score:.4f}")
        if score >= threshold:
            print(f"⚠️ Duplicate found: {user_id} (Score: {score:.4f})")
            found = True

    if not found:
        print("✅ No duplicate users detected.")
        # Add to dataframe
        new_row = pd.DataFrame([{
        "user_id": os.path.basename(new_user_img_path),
        "embedding": new_embedding
        }])
        df = pd.concat([df, new_row], ignore_index=True)

        # Save back to pickle
        with open(embedding_file, "wb") as f:
            pickle.dump(df.to_dict(orient="list"), f)
        print("📦 New user embedding added to database.")

# -----------------------------
# Run the duplicate check for an input image
if __name__ == "__main__":
    if not os.path.exists(embedding_file):
        extract_and_store_embeddings()
    else:
        print("✅ Loaded existing embeddings.")

    df = load_embeddings()

    if not os.path.exists(input_img_path):
        print(f"❌ Input image '{input_img_path}' not found!")
    else:
        check_duplicate(input_img_path, df)
