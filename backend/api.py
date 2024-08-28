from flask import Flask, request, jsonify
import numpy as np
import requests
import pickle
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
from flask_cors import CORS
from main import get_image_embedding

load_dotenv()

# Get API key from environment variable .env
LIMEWIRE_API_KEY = os.getenv("LIMEWIRE_API_KEY")

app = Flask(__name__)
CORS(app)

curr_dir = os.path.dirname(os.path.realpath(__file__))

# Load the embeddings database
with open(curr_dir + "/embeddings_database.pkl", "rb") as f:
    embeddings_database = pickle.load(f)


def calculate_similarity(embedding1, embedding2):
    """Calulates cosine similarity between two embeddings
    If cosine similarity is 1, the embeddings are identical
    If cosine similarity is 0, the embeddings are orthogonal
    If cosine similarity is -1, the embeddings are opposite"""

    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()

    dot_product = np.dot(embedding1, embedding2)

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    return dot_product / (norm1 * norm2)


def find_most_similar_logo(new_logo_embedding, embeddings_database):
    """Comapres the new logo with logos in the database"""
    max_similarity = -1
    most_similar_logo = None

    for filename, embedding in embeddings_database.items():
        similarity = calculate_similarity(new_logo_embedding, embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_logo = filename

    return most_similar_logo, max_similarity


@app.route("/compare-logo", methods=["POST"])
def compare_logo():
    """Route to compare logo with logos in database"""

    file = request.files["file"]
    image = Image.open(BytesIO(file.read()))

    new_logo_embedding = get_image_embedding(image)

    # Find the most similar logo in the database
    most_similar_logo, similarity_score = find_most_similar_logo(
        new_logo_embedding, embeddings_database
    )
    
    print(most_similar_logo)

    return jsonify(
        {
            "most_similar_logo": most_similar_logo,
            "similarity_score": str(round(float(similarity_score) * 100, 2)) + '%',
        }
    )


@app.route("/generate-logo", methods=["POST"])
def generate_logo():
    """Route to generate logo from description
    https://docs.limewire.com/#operation/generateImage"""

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
    """Default route"""
    return "Welcome to the logo trademark API!"


if __name__ == "__main__":
    app.run(debug=True)
