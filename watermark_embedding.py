import numpy as np
import pywt
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io

arrign=[]
def perform_lwt(image, levels):
    """Perform Lifting Wavelet Transform (LWT) on the original image"""
    coeffs = image.copy()
    for _ in range(levels):
        coeffs,ign = pywt.dwt2(coeffs, 'haar')  # Perform LWT using Haar wavelet
        arrign.append(ign)
    hl3_subband = coeffs
    return hl3_subband

def inverse_lwt(subband, image_shape):
    """Perform inverse LWT to obtain watermarked image"""
    coeffs = subband
    for i in range(3):
        coeffs = pywt.idwt2((coeffs,arrign[-(i+1)]), 'haar')
    watermarked_image = coeffs
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

def embed_watermark(original_image, reference_watermark, signature_watermark, key1, key2, embedding_threshold):
    # Step 1: Perform LWT three times to obtain HL3 sub-band
    hl3_subband = perform_lwt(original_image, 3)
    # # Step 2: Randomize coefficients
    randomized_subband = randomize_coefficients(hl3_subband, key1,True)
    # return randomized_subband
    # # Step 3: Arrange and scramble coefficients
    scrambled_subband = arrange_and_scramble(randomized_subband, key2 ,True)
    # # Step 4: Perform SVD on every block
    scrambled_subband = scrambled_subband.reshape(-1, 2, 2)
    singular_matrices = perform_svd(scrambled_subband)
    # # # Step 5: Calculate average difference between singular values
    average_difference = calculate_average_difference(singular_matrices)
    # return average_difference
    # # Step 6: Concatenate reference and signature watermarks
    watermark = np.concatenate((reference_watermark, signature_watermark))
    
    # # Step 7: Embed watermark
    for bit in watermark:
        for matrix in singular_matrices:
            if bit == 1:
                dominant_index = np.argmax(matrix[1])
                dominant_value = matrix[1][dominant_index]
                threshold = embedding_threshold if dominant_value < average_difference else average_difference
                matrix[1][dominant_index] += threshold
            # For bit 0, no modification needed

    # Step 8: Reconstruct blocks using modified singular matrices
    modified_subband = np.array([np.dot(np.dot(u, np.diag(s)), vh) for u, s, vh in singular_matrices])
    # Step 9: Inverse shuffle coefficients
    inverse_shuffled_subband = arrange_and_scramble(modified_subband.reshape(-1,len(hl3_subband)), key2,False)
    inverse_shuffled_subband = randomize_coefficients(inverse_shuffled_subband,key1,False)
    # Step 10: Perform inverse LWT
    watermarked_image = inverse_lwt(inverse_shuffled_subband, original_image.shape)
    
    return watermarked_image

# Main script
reference_watermark = np.array([1,0,0,1,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,1,1,1,1,0,0,0,0,0,1,1,0,1,0,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,1,0,0,0,1,1,1,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1])
signature_watermark = np.array([1,0,1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,0,1,0,0,1,1,0,1,0,1,0,1,0,1,1,1,1,0,1,0,1,1,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,1])
def embedding(var,name):
    # Load the original image
    coloured_image = cv2.imread(var)
    coloured_image = cv2.resize(coloured_image, (512, 512))
    original_image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)
    # Generate reference and signature watermarks
    # plt.imsave("GreyScale/"+name+"-greyscale.png",original_image,cmap='grey')

    # Define secret seed keys
    key1 = 123
    key2 = 234

    # Define embedding threshold
    embedding_threshold = 0.5

    # Embed watermark
    watermarked_image = embed_watermark(original_image, reference_watermark, signature_watermark, key1, key2, embedding_threshold)
    # # Display the watermarked image
    # plt.imshow(watermarked_image, cmap='gray')
    # plt.axis('off')
    # plt.title('Watermarked Image')
    # plt.show()

    # Save the watermarked image
    output_path = "embedded/"+name.split('.')[0]+'-embedded.png'
    plt.imsave(output_path, watermarked_image, cmap='gray')

for image_file in os.listdir("non-embedded-train"):
    image_path=os.path.join("non-embedded-train",image_file)
    embedding(image_path,image_file)