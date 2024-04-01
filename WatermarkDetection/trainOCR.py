import cv2
import os
import numpy as np
from sklearn.svm import SVC
from skimage.feature import graycomatrix, graycoprops
import joblib
import pytesseract

# Function to extract features from an image
def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocess image to enhance text visibility (e.g., thresholding, smoothing)
    # Example: Thresholding
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Perform OCR (Optical Character Recognition) to extract text
    text = pytesseract.image_to_string(binary_image)
    
    # Example: Calculate the number of characters extracted
    num_characters = len(text)
    
    # Example: Calculate other features relevant to extracted text
    
    # Combine features into a feature vector
    features = np.array([num_characters])
    
    return features

# Paths to the folders containing watermarked and non-watermarked images
watermarked_folder = 'wm-nowm/train/watermark'
non_watermarked_folder = 'wm-nowm/train/no-watermark'

# Load watermarked images
watermarked_images = []
limit = 0
batch_size=100000
for filename in os.listdir(watermarked_folder):
    path = os.path.join(watermarked_folder, filename)
    image = cv2.imread(path)
    if image is not None:
        watermarked_images.append(image)
    else:
        print(f"Error: Unable to load the image at {path}")
    limit += 1
    if limit >= batch_size:
        break

print(f"Number of watermarked images loaded: {len(watermarked_images)}")

# Load non-watermarked images
limit = 0
non_watermarked_images = []
for filename in os.listdir(non_watermarked_folder):
    path = os.path.join(non_watermarked_folder, filename)
    image = cv2.imread(path)
    if image is not None:
        non_watermarked_images.append(image)
    else:
        print(f"Error: Unable to load the image at {path}")
    limit += 1
    if limit >= batch_size:
        break

print(f"Number of non-watermarked images loaded: {len(non_watermarked_images)}")


# Extract features and assign labels
X = []
y = []

# For watermarked images, label = 1
for image in watermarked_images:
    features = extract_features(image)
    X.append(features)
    y.append(1)

# For non-watermarked images, label = 0
for image in non_watermarked_images:
    features = extract_features(image)
    X.append(features)
    y.append(0)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Train SVM
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X, y)

# Save the trained model
joblib.dump(svm, 'watermark_detection_modelOCR.pkl')
