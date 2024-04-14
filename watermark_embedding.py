import numpy as np
import pywt

# Step 1: Perform LWT on the original image
def perform_lwt(image, levels):
    coeffs = pywt.wavedec2(image, 'haar', level=levels)
    hl3_subband = coeffs[1]  # Select the HL3 sub-band
    return hl3_subband

# Step 2: Randomize the coefficients
def randomize_coefficients(subband, seed):
    np.random.seed(seed)
    return np.random.permutation(subband)

# Step 3: Arrange coefficients into blocks and scramble them
def arrange_and_scramble(coefficients, seed):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(coefficients))
    return coefficients[shuffled_indices].reshape(coefficients.shape)

# Step 4: Perform SVD on each block
def perform_svd(blocks):
    singular_matrices = []
    for block in blocks:
        _, singular_values, _ = np.linalg.svd(block)
        singular_matrices.append(np.diag(singular_values))
    return singular_matrices

# Step 5: Calculate average difference between singular values
def calculate_average_difference(singular_matrices):
    differences = []
    for matrix in singular_matrices:
        differences.append(np.max(matrix) - np.min(matrix))
    return np.mean(differences)

# Step 6: Concatenate reference and signature watermarks
def concatenate_watermark(reference, signature):
    return np.concatenate((reference, signature))

# Step 7: Modify singular values based on watermark bits
# Step 7: Modify singular values based on watermark bits
def modify_singular_values(singular_matrices, bit, threshold):
    for matrix in singular_matrices:
        if bit == 1:
            # Get the index of the dominant singular value
            dominant_index = np.argmax(matrix)
            # Check if dominant_index exceeds the size of the matrix
            if dominant_index < len(matrix):
                # Modify the dominant singular value based on the threshold
                matrix[dominant_index] -= threshold
        # For bit 0, no modification needed


# Step 8: Reconstruct blocks using modified singular matrices
def reconstruct_blocks(singular_matrices):
    reconstructed_blocks = []
    for matrix in singular_matrices:
        reconstructed_blocks.append(np.dot(np.dot(matrix, matrix.T), matrix))
    return np.array(reconstructed_blocks)

# Step 9: Inverse shuffle the blocks and coefficients
def inverse_shuffle(coefficients, seed):
    np.random.seed(seed)
    original_indices = np.argsort(np.random.permutation(len(coefficients)))
    return coefficients[original_indices]

# Step 10: Perform inverse LWT to obtain watermarked image
def perform_inverse_lwt(subband, levels):
    watermarked_image = pywt.waverec2([subband], 'haar')
    return watermarked_image

# Watermark embedding function
# def embed_watermark(original_image, reference_watermark, signature_watermark, key1, key2, embedding_threshold):
#     # Step 1: Perform LWT
#     hl3_subband = perform_lwt(original_image, 3)

#     # Step 2: Randomize coefficients
#     randomized_subband = randomize_coefficients(hl3_subband, key1)

#     # Step 3: Arrange and scramble coefficients
#     scrambled_blocks = arrange_and_scramble(randomized_subband, key2)

#     # Step 4: Perform SVD
#     singular_matrices = perform_svd(scrambled_blocks)

#     # Step 5: Calculate average difference
#     average_difference = calculate_average_difference(singular_matrices)

#     # Step 6: Concatenate watermarks
#     watermark = concatenate_watermark(reference_watermark, signature_watermark)

#     # Step 7: Modify singular values
#     for bit in watermark:
#         modify_singular_values(singular_matrices, bit, embedding_threshold)

#     # Step 8: Reconstruct blocks
#     reconstructed_blocks = reconstruct_blocks(singular_matrices)

#     # Step 9: Inverse shuffle
#     inverse_shuffled_blocks = inverse_shuffle(reconstructed_blocks, key2)

#     # Step 10: Inverse LWT
#     watermarked_image = perform_inverse_lwt(inverse_shuffled_blocks, 3)

#     return watermarked_image

# Watermark embedding function
def embed_watermark(original_image, reference_watermark, signature_watermark, key1, key2, embedding_threshold):
    # Convert seed keys to integers within the valid range
    key1 = abs(hash(key1)) % (2**32 - 1)
    key2 = abs(hash(key2)) % (2**32 - 1)

    # Step 1: Perform LWT
    hl3_subband = perform_lwt(original_image, 3)

    # Step 2: Randomize coefficients
    randomized_subband = randomize_coefficients(hl3_subband, key1)

    # Step 3: Arrange and scramble coefficients
    scrambled_blocks = arrange_and_scramble(randomized_subband, key2)

    # Step 4: Perform SVD
    singular_matrices = perform_svd(scrambled_blocks)

    # Step 5: Calculate average difference
    average_difference = calculate_average_difference(singular_matrices)

    # Step 6: Concatenate watermarks
    watermark = concatenate_watermark(reference_watermark, signature_watermark)

    # Step 7: Modify singular values
    for bit in watermark:
        modify_singular_values(singular_matrices, bit, embedding_threshold)

    # Step 8: Reconstruct blocks
    reconstructed_blocks = reconstruct_blocks(singular_matrices)

    # Step 9: Inverse shuffle
    inverse_shuffled_blocks = inverse_shuffle(reconstructed_blocks, key2)

    # Step 10: Inverse LWT
    watermarked_image = perform_inverse_lwt(inverse_shuffled_blocks, 3)

    return watermarked_image

# # Example usage:
# original_image = np.random.rand(512, 512)  # Example original image
# reference_watermark = np.random.randint(2, size=448)  # Example reference watermark
# signature_watermark = np.random.randint(2, size=64)  # Example signature watermark
# key1 = 'random_seed1'  # Example secret seed key1
# key2 = 'random_seed2'  # Example secret seed key2
# embedding_threshold = 0.5  # Example embedding threshold

# watermarked_image = embed_watermark(original_image, reference_watermark, signature_watermark, key1, key2, embedding_threshold)

# Example usage:
original_image = np.random.rand(512, 512)  # Example original image
reference_watermark = np.random.randint(2, size=448)  # Example reference watermark
signature_watermark = np.random.randint(2, size=64)  # Example signature watermark
key1 = 'random_seed1'  # Example secret seed key1
key2 = 'random_seed2'  # Example secret seed key2
embedding_threshold = 0.5  # Example embedding threshold

# Print original image shape and first few pixels
print("Original image shape:", original_image.shape)
print("First few pixels of the original image:\n", original_image[:3, :3])

# Embed watermark
watermarked_image = embed_watermark(original_image, reference_watermark, signature_watermark, key1, key2, embedding_threshold)

# Print watermarked image shape and first few pixels
print("\nWatermarked image shape:", watermarked_image.shape)
print("First few pixels of the watermarked image:\n", watermarked_image[:3, :3])
