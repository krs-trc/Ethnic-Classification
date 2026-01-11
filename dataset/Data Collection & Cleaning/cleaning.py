import os
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image

# Initialize MTCNN face detector
detector = MTCNN()

# Define input and output directories
input_folder = "dataset_cleaning_input"
output_folder = "cleaned_image"
os.makedirs(output_folder, exist_ok=True)

def is_blurry(image, threshold=100):
    """Check if an image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold  # Lower variance means blurrier image

def process_images():
    """Filter and preprocess images."""
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)

        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipping {filename} (not an image)")
                continue

            # Detect faces
            faces = detector.detect_faces(image)
            if len(faces) != 1:  # Skip if no face or multiple faces
                print(f"Skipping {filename} (no face or multiple faces)")
                continue

            # Check image quality (blurriness)
            if is_blurry(image):
                print(f"Skipping {filename} (blurry image)")
                continue

            # Crop and align the face
            x, y, w, h = faces[0]['box']
            face = image[y:y+h, x:x+w]

            # Resize to 224x224 (for VGGFace compatibility)
            face_resized = cv2.resize(face, (512, 512))

            # Convert to RGB and save
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, face_resized)
            print(f"Saved cleaned image: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Run the preprocessing
process_images()
