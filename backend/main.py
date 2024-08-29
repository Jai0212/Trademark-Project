import tensorflow as tf
import numpy as np
import pickle
from datasets import load_dataset
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))
images_dir = os.path.join(curr_dir, "images")  # Directory to store images

# Ensure the images directory exists
os.makedirs(images_dir, exist_ok=True)

# Load sample dataset
ds = load_dataset("samp3209/logo-dataset")

# Initialize the ResNet50 CNN model
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


def create_embeddings_database(dataset):
    """Creates the dataset with embeddings and stores images in RGB mode"""
    embeddings = {}

    for index, example in enumerate(dataset["train"]):  # index is uid, example is dict
        image = example["image"]
        
        # Convert image to RGB mode if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate embedding
        embedding = get_image_embedding(image)
        
        # Save image to a specified path
        image_path = os.path.join(images_dir, f"image_{index}.png")
        image.save(image_path)

        # Store embedding and image path
        embeddings[f"image_{index}"] = {
            "embedding": embedding,
            "image_path": image_path
        }

    # Save embeddings to a file
    with open(os.path.join(curr_dir, "embeddings_database.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":
    create_embeddings_database(ds)
