import numpy as np
import pywt
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
features = []
label = []
def psnr(original, compressed):
    """
    Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value


def ncc(image1, image2):
    """
    Compute Normalized Cross-Correlation (NCC) between two images.
    """
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    mean_image1 = np.mean(image1_flat)
    mean_image2 = np.mean(image2_flat)
    numerator = np.sum((image1_flat - mean_image1)
                       * (image2_flat - mean_image2))
    denominator = np.sqrt(np.sum((image1_flat - mean_image1) ** 2)
                          * np.sum((image2_flat - mean_image2) ** 2))
    ncc_value = numerator / denominator
    return ncc_value


def extract_features(HL3, reference_watermark, signature_watermark, key1, key2, embedding_threshold):
    # print("Step 1: Perform LWT three times to obtain HL3 sub-band")
    hl3_subband = HL3

    # print("Step 2: Randomize coefficients")
    randomized_subband = randomize_coefficients(hl3_subband, key1, True)

    # print("Step 3: Arrange and scramble coefficients")
    scrambled_subband = arrange_and_scramble(randomized_subband, key2, True)

    # print("Step 4: Perform SVD on every block")
    scrambled_subband = scrambled_subband.reshape(-1, 2, 2)
    singular_matrices = perform_svd(scrambled_subband)
    arr=[]
    for matrix in singular_matrices:
        arr.append(matrix[1])
    features=[
            # np.mean(arr),                # Mean (p1)
            np.var(arr),                 # Variance (p2)
            # np.std(arr),                 # Standard deviation (p3)
            # np.median(arr),              # Median (p4)
            # #np.cov(matrix[1]),                 # Covariance (p5)
            # np.mean(np.power(arr, 5)),   # Moment (5th order) (p6)
            # np.percentile(arr, q=75),    # Quantile (p7)
            # #np.diff(matrix[1]).tolist(),       # Difference between singular values (p8)
            # #np.diff(np.square(matrix[1])).tolist(),  # Difference between the square of singular values (p9)
            # np.sum(np.square(arr))      # Energy (p10)
        ]
    return np.array(features)


def perform_lwt(image):
    image = image.copy()

    # Perform wavelet transform
    coeffs = pywt.wavedec2(image, 'haar', level=3)
    # print(coeffs)

    return coeffs


def inverse_lwt(all_coeffs):
    # Perform inverse wavelet transform
    watermarked_image = pywt.waverec2(all_coeffs, 'haar')

    return watermarked_image

def randomize_coefficients(subband, key1,var):
    """Randomize HL3 sub-band coefficients using secret seed key1"""
    np.random.seed(key1)
    if var==False:
        return np.argsort(np.random.permutation(subband))
    return np.random.permutation(subband)

def arrange_and_scramble(coefficients, key2,var):
    """Arrange and scramble coefficients in non-overlapping 2x2 blocks using secret seed key2"""
    np.random.seed(key2)
    shuffled_indices = np.random.permutation(coefficients.shape[0] * coefficients.shape[1])
    if var==False:
        shuffled_indices = np.argsort(shuffled_indices)
    return coefficients.reshape(-1)[shuffled_indices].reshape(coefficients.shape)

def perform_svd(blocks):
    """Perform Singular Value Decomposition (SVD) on every block"""
    singular_matrices = []
    for block in blocks:
        u, s, vh = np.linalg.svd(block, full_matrices=False)
        singular_matrices.append((u, s, vh))
    return singular_matrices

def calculate_average_difference(singular_matrices):
    """Calculate average difference between singular values"""
    differences = []
    for matrix in singular_matrices:
        differences.append(np.max(matrix[1]) - np.min(matrix[1]))
    return np.mean(differences)

reference_watermark = np.array([1,0,0,1,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,1,1,1,1,0,0,0,0,0,1,1,0,1,0,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,1,0,0,0,1,1,1,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1])
signature_watermark = np.array([1,0,1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,0,1,0,0,1,1,0,1,0,1,0,1,0,1,1,1,1,0,1,0,1,1,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,1])
features = []
label =[]
def operate(var):
    # Load the original image
    coloured_image = cv2.imread(var)
    coloured_image = cv2.resize(coloured_image, (512, 512))
    original_image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)
    

    # Define secret seed keys
    key1 = 123
    key2 = 234

    # Define embedding threshold
    embedding_threshold = 0.5
    # print('perform lwt on ', image_file)
    coeffs = perform_lwt(original_image)
    LL, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs
    # print('HL3 ', HL3)  # 64x64
    feat = extract_features(
        HL3, reference_watermark, signature_watermark, key1, key2, embedding_threshold)
    # Embed watermark
    # feat = extract_features(original_image, reference_watermark, signature_watermark, key1, key2, embedding_threshold)
    return feat
for image_file in os.listdir("non-embedded-train"):
    image_path=os.path.join("non-embedded-train",image_file)
    features.append(operate(image_path))
    label.append(0)

for image_file in os.listdir("embedded-train-advay"):
    image_path=os.path.join("embedded-train-advay",image_file)
    features.append(operate(image_path))
    label.append(1)

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.1, random_state=42)

# file = open("output.txt",'w')
# for i in range(len(X_train)):
#     file.write(str(X_train[i]))
#     file.write(" : ")
#     file.write(str(y_train[i]))
#     file.write("\n")
# file.close()
# # Initialize SVM classifier
clf = svm.SVC(kernel='rbf')

# # Train the SVM classifier
clf.fit(X_train, y_train)
# # Evaluate the model

# # Save the trained model
# joblib.dump(clf, 'svm_model.pkl')

# print("SVM model saved.")
predictions = clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Print accuracy and classification report
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

