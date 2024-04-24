import numpy as np
import pywt
import cv2
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib

np.set_printoptions(threshold=np.inf)

def perform_lwt(image, levels):
    """Perform Lifting Wavelet Transform (LWT) on the original image"""
    # Define embedding threshold
    coeffs = pywt.wavedec2(image, 'haar', level=3)
    LL, (HL3, LH3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs

    return HL3

def randomize_coefficients(subband, key1,var):
    """Randomize HL3 sub-band coefficients using secret seed key1"""
    np.random.seed(key1)
    shuffled_indices = np.random.permutation(subband.shape[0] * subband.shape[1])
    if var==False:
        shuffled_indices = np.argsort(shuffled_indices)
    return subband.reshape(-1)[shuffled_indices].reshape(subband.shape)

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

def extract_features(singular_values_matrix):
    features = []
    for sv_matrix in singular_values_matrix:
        diff = np.max(sv_matrix[1])-np.min(sv_matrix[1])
        sv_flat = sv_matrix[1].flatten()
        mean_val = np.mean(sv_flat)  
        variance_val = np.var(sv_flat)  
        std_dev_val = np.std(sv_flat)  
        median_val = np.median(sv_flat)  
        #covariance_val = np.cov(sv_flat.reshape(-1, 1))[0, 0]
        moment_val = np.mean(np.power(sv_flat, 5))  
        quantile_val = np.percentile(sv_flat, q=75)  
        diff_singular_vals = np.diff(sv_flat)[0]
        diff_sq_singular_vals = np.diff(np.square(sv_flat))[0]
        energy_val = np.sum(np.square(sv_flat))  
        
        largest_singular_vals = np.sort(sv_flat)[-2:]

        # Append computed features and singular values to the feature list
        features.append([diff])

    return np.array(features)

# Main script
reference_watermark = np.array([1,0,0,1,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,1,1,1,1,0,0,0,0,0,1,1,0,1,0,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,1,0,0,0,1,1,1,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1])
signature_watermark = np.array([1,0,1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,0,1,0,0,1,1,0,1,0,1,0,1,0,1,1,1,1,0,1,0,1,1,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,1])

features = []
label =[]

def operate(image_path):
    '''This function would get the image path and return the predicted watermark'''
    print(image_path)
    coloured_image = cv2.imread(image_path)
    coloured_image = cv2.resize(coloured_image, (512, 512))
    original_image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)

    key1 = 123
    key2 = 234

    hl3_subband = perform_lwt(original_image, 3)

    randomized_subband = randomize_coefficients(hl3_subband, key1, True)

    scrambled_subband = arrange_and_scramble(randomized_subband, key2, True)

    scrambled_subband = scrambled_subband.reshape(-1, 2, 2)
        
    singular_matrices = perform_svd(scrambled_subband)

    extracted_features = extract_features(singular_matrices)
    X_train = extracted_features[:512]
    X_test = extracted_features[512:]
    Y_train = reference_watermark 

    clf = svm.SVC(kernel='rbf')

    clf.fit(X_train, Y_train)

    # Predict SW using SVM classifier
    predicted_sw = clf.predict(X_test)

    return predicted_sw



for image_file in os.listdir("embedded-train-new-aman"):
    image_path=os.path.join("embedded-train-new-aman",image_file)
    predicted_watermark = operate(image_path)
    count = 0
    for index, value in enumerate(predicted_watermark): 
        if value == signature_watermark[index]:
            count += 1

    accuracy = count / 512
    print("Accuracy is:", accuracy)
