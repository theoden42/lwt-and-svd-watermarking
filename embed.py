import numpy as np
import pywt
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io

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


def generate_watermark_HL3(HL3, reference_watermark, signature_watermark, key1, key2, embedding_threshold):
    print("Step 1: Perform 3 level LWT to obtain HL3 sub-band")
    hl3_subband = HL3
    print("Step 2: Randomize coefficients")
    randomized_subband = randomize_coefficients(hl3_subband, key1, True)
    print("Step 3: Arrange and scramble blocks")
    scrambled_subband = arrange_and_scramble(randomized_subband, key2, True)

    print("Step 4: Perform SVD on every block")
    scrambled_subband = scrambled_subband.reshape(-1, 2, 2)
    singular_matrices = perform_svd(scrambled_subband)
    print("Step 5: Calculate average difference between singular values")
    average_difference = calculate_average_difference(singular_matrices)

    print("Step 6: Concatenate reference and signature watermarks")
    watermark = np.concatenate((reference_watermark, signature_watermark))
    print(watermark)
    print("Step 7: Embed watermark")
    # for bit in watermark:
    #     for matrix in singular_matrices:
    #         dominant_index = np.argmax(matrix[1])
    #         if bit == 1:
    #             threshold = embedding_threshold if (np.max(matrix[1]) - np.min(matrix[1])) < average_difference else 0
    #             matrix[1][dominant_index] += threshold
            # else :
            #     matrix[1][dominant_index]=matrix[1][1-dominant_index]
    # for bit in watermark:
    #     for matrix in singular_matrices:
    #         if bit == 1:
    #             dominant_index = np.argmax(matrix[1])
    #             dominant_value = matrix[1][dominant_index]
    #             threshold = embedding_threshold if dominant_value < average_difference else average_difference
    #             matrix[1][dominant_index] += threshold

    print("Step 8: Reconstruct blocks using modified singular matrices")
    modified_subband = np.array(
        [np.dot(np.dot(u, np.diag(s)), vh) for u, s, vh in singular_matrices])
    print("Step 9: Inverse shuffle coefficients")
    inverse_shuffled_subband = arrange_and_scramble(
        modified_subband.reshape(-1, len(hl3_subband)), key2, False)
    inverse_shuffled_subband = randomize_coefficients(inverse_shuffled_subband, key1, False)
    print("Watermark embedded successfully.")
    return inverse_shuffled_subband


def perform_lwt(image):
    image = image.copy()

    # Perform wavelet transform
    coeffs = pywt.wavedec2(image, 'haar', level=3)
    print(coeffs)

    return coeffs


def inverse_lwt(all_coeffs):
    # Perform inverse wavelet transform
    watermarked_image = pywt.waverec2(all_coeffs, 'haar')

    return watermarked_image

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

def calculate_average_difference(singular_matrices):
    """Calculate average difference between singular values"""
    differences = []
    for matrix in singular_matrices:
        differences.append(np.max(matrix[1]) - np.min(matrix[1]))
    return np.mean(differences)


def generate_sync_info(image):
    # Calculate invariant centroid
    moments = cv2.moments(image)
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']

    # Calculate size of the image
    image_size = image.shape[0] * image.shape[1]

    # Calculate Radon transform phase information
    theta = np.rad2deg(np.arctan2(image.shape[1], image.shape[0]))

    # Convert to binary strings
    centroid_binary = format(int(cx), '024b') + format(int(cy), '024b')
    size_binary = format(image_size, '024b')
    theta_binary = format(int(theta), '016b')

    # Combine all information into a single 64-bit string
    sync_info = centroid_binary + size_binary + theta_binary

    return sync_info


# def generate_rw(image, key1):
#     # Generate random bits as string array
#     random_bits = np.random.randint(0, 2, size=448).astype(str)

#     # Generate synchronization information
#     sync_info = generate_sync_info(image)

#     # Concatenate random bits and sync info
#     reference_watermark = np.concatenate((random_bits, [sync_info]))

#     return reference_watermark


reference_watermark = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0,
                              0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
signature_watermark = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                               1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1])

key1 = 123
key2 = 234

embedding_threshold = 0.5


def embed_watermark(img_path):
    original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image, (512, 512))

    print('perform lwt on ', image_file)
    coeffs = perform_lwt(original_image)
    LL, (HL3, LH3, HH3), (HL2, LH2, HH2), (HL1, LH1, HH1) = coeffs
    print('HL3 ', HL3)  # 64x64

    # reference_watermark = generate_rw(original_image, key1)

    new_HL3 = generate_watermark_HL3(
        HL3, reference_watermark, signature_watermark, key1, key2, embedding_threshold)

    print('new HL3 ', new_HL3)
    new_coeffs = LL, (new_HL3, LH3, HH3), (HL2, LH2, HH2), (HL1, LH1, HH1)

    watermarked_image = inverse_lwt(new_coeffs)

    new_image_name = image_file.split(".")[0] + "_embedded.png"
    new_path = os.path.join("embedded-train-new-aman",
                            new_image_name)
    cv2.imwrite(new_path, watermarked_image)

    psnr_value = psnr(original_image, watermarked_image)
    print("PSNR:", psnr_value)

    ncc_value = ncc(original_image, watermarked_image)
    print("NCC:", ncc_value)

    return new_path


for image_file in os.listdir("non-embedded-train"):
    image_path = os.path.join("non-embedded-train", image_file)
    embed_watermark(image_path)
    break
    # print(generate_sync_info(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)))