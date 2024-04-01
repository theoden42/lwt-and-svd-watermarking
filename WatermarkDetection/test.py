import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from skimage.feature import graycomatrix, graycoprops
import joblib
import pytesseract
# Function to extract features from an image
def extract_features(image):
    # Example: Extract texture features using GLCM (Gray-Level Co-occurrence Matrix)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    features = np.hstack([contrast, energy, correlation])
    return features

def extract_edge(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    mean_edge_intensity = np.mean(edges)
    num_edges = np.sum(edges)
    features = np.array([mean_edge_intensity, num_edges])
    return features

def extract_features_OCR(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary_image)
    num_characters = len(text)
    features = np.array([num_characters])
    return features

# Load the trained model
# Change to appropriate model
svm = joblib.load('watermark_detection_modelOCR.pkl')

watermarked_folder = 'wm-nowm/valid/watermark'
non_watermarked_folder = 'wm-nowm/valid/no-watermark'

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
X_test = []
y_test = []

# For watermarked images, label = 1
for image in watermarked_images:
    features = extract_features_OCR(image)
    X_test.append(features)
    y_test.append(1)

# For non-watermarked images, label = 0
for image in non_watermarked_images:
    features = extract_features_OCR(image)
    X_test.append(features)
    y_test.append(0)

# Convert lists to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)
# Predict on test set
predictions = svm.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Print accuracy and classification report
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# For basic features

# with open('classification_report.txt', 'w') as file:
#     file.write("Accuracy: " + str(accuracy) + "\n")
#     file.write("Classification Report:\n" + classification_rep)

# For Edge Detection and Text Localiztion
 
# with open('classification_report_Edge.txt', 'w') as file:
#     file.write("Accuracy: " + str(accuracy) + "\n")
#     file.write("Classification Report:\n" + classification_rep)

# For OCR

with open('classification_report_OCR2.txt', 'w') as file:
    file.write("Accuracy: " + str(accuracy) + "\n")
    file.write("Classification Report:\n" + classification_rep)