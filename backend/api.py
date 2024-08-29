from flask import Flask, request, jsonify
import numpy as np
import requests
import pickle
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
from flask_cors import CORS
import base64
import requests
import tensorflow as tf
import numpy as np

load_dotenv()

# Get API key from environment variable .env
LIMEWIRE_API_KEY = os.getenv("LIMEWIRE_API_KEY")

app = Flask(__name__)
CORS(app)

curr_dir = os.path.dirname(os.path.realpath(__file__))

# Load the embeddings database
with open(os.path.join(curr_dir, "embeddings_database.pkl"), "rb") as f:
    embeddings_database = pickle.load(f)


model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, pooling="avg"
)


def preprocess_image(image):
    """Preprocess the image"""
    # Convert image to numpy array
    image = np.array(image)

    if image.ndim == 2:  # if image is greyscale (2D), add channel dimension
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)  # Convert greyscale to RGB

    if (
        image.ndim == 3 and image.shape[-1] == 1
    ):  # If image is greyscale (3D with 1 channel)
        image = np.repeat(image, 3, axis=-1)  # Convert greyscale to RGB

    # Finalize image dimensions
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    image = tf.image.resize(image, [224, 224])  # Resize image
    image = tf.keras.applications.resnet50.preprocess_input(image)

    return image


def get_image_embedding(image):
    """Get the image embedding"""
    preprocessed_image = preprocess_image(image)
    embedding = model.predict(preprocessed_image)
    return embedding


def calculate_similarity(embedding1, embedding2):
    """Calculates cosine similarity between two embeddings."""
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()

    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    return dot_product / (norm1 * norm2)


def find_most_similar_logo(new_logo_embedding, embeddings_database):
    """Compares the new logo with logos in the database."""
    max_similarity = -1
    most_similar_logo = None

    for filename, data in embeddings_database.items():
        embedding = data["embedding"]  # Access the embedding part
        similarity = calculate_similarity(new_logo_embedding, embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_logo = filename

    return most_similar_logo, max_similarity


@app.route("/compare-logo", methods=["POST"])
def compare_logo():
    """Route to compare logo with logos in database."""

    file = request.files["file"]
    image = Image.open(BytesIO(file.read()))

    new_logo_embedding = get_image_embedding(image)

    # Find the most similar logo in the database
    most_similar_logo, similarity_score = find_most_similar_logo(
        new_logo_embedding, embeddings_database
    )

    # Load the most similar logo image from the filesystem
    similar_logo_path = os.path.join(curr_dir, "images/", most_similar_logo + ".png")
    with open(similar_logo_path, "rb") as img_file:
        # Encode the image to base64
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    return jsonify(
        {
            "most_similar_logo": most_similar_logo,
            "similarity_score": str(round(float(similarity_score) * 100, 2)) + "%",
            "image_base64": img_base64,
        }
    )


@app.route("/generate-logo", methods=["POST"])
def generate_logo():
    """Route to generate logo from description."""
    prompt = request.json.get("prompt")

    url = "https://api.limewire.com/api/image/generation"

    payload = {"prompt": prompt, "aspect_ratio": "1:1"}

    headers = {
        "Content-Type": "application/json",
        "X-Api-Version": "v1",
        "Accept": "application/json",
        "Authorization": "Bearer " + LIMEWIRE_API_KEY,
    }

    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    if response.status_code == 200 and data.get("status") == "COMPLETED":
        asset_url = data["data"][0]["asset_url"]
        return jsonify({"image_url": asset_url})
    else:
        return jsonify({"error": "Failed to generate image"}), response.status_code


@app.route("/")
def home():
    """Default route."""
    return "Welcome to the logo trademark API!"


if __name__ == "__main__":
    app.run(debug=True)
