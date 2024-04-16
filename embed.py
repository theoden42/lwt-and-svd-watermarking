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

    print('embed in ', img_path)

    coeffs = perform_lwt(original_image)
    LL, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs
    new_HL3 = HL3  # temporary, using same HL3
    new_coeffs = LL, (LH3, new_HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1)

    watermarked_image = inverse_lwt(new_coeffs)

    new_image_name = img_path.split(".")[0] + "_embedded.png"
    new_path = os.path.join("embedded-train-advay",
                            new_image_name)
    cv2.imwrite(new_path, watermarked_image)

    psnr_value = psnr(original_image, watermarked_image)
    psnr_list.append(psnr_value)
    print("PSNR:", psnr_value)

    ncc_value = ncc(original_image, watermarked_image)
    ncc_list.append(ncc_value)
    print("NCC:", ncc_value)

    return new_path


psnr_list = []
ncc_list = []
for image_file in os.listdir("non-embedded-train"):
    image_path = os.path.join("non-embedded-train", image_file)
    embed_watermark(image_path)
    # print(generate_sync_info(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)))

print('For ', len(os.listdir("non-embedded-train")),
      ' sample images: PSNR and NCC values have:')
print("Minimum:", np.min(psnr_list),
      "| Maximum:", np.max(psnr_list),
      "| Mean:", np.mean(psnr_list),
      "| Standard Deviation:", np.std(psnr_list))

print("Minimum:", np.min(ncc_list), "| Maximum:", np.max(ncc_list),
      "| Mean:", np.mean(ncc_list), "| Standard Deviation:", np.std(ncc_list))
