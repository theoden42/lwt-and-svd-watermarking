import cv2
import numpy as np
import pywt

# Read host image
host_img = cv2.imread('host_image.jpg', cv2.IMREAD_GRAYSCALE)

coeffs = pywt.dwt2(host_img, 'haar')
cA, (cH, cV, cD) = coeffs

# Embedding the watermark into the approximation coefficients
watermark = np.random.randint(0, 2, size=cA.shape)  # Example watermark, you can replace it with your own
alpha = 0.1  # Strength of watermark
watermarked_cA = cA + alpha * watermark

# Perform inverse DWT
watermarked_img = pywt.idwt2((watermarked_cA, (cH, cV, cD)), 'haar')

# Perform SVD on the watermarked approximation coefficients
U, S, Vt = np.linalg.svd(watermarked_cA)

# Extracting the watermark
# Reconstruct approximation coefficients with inverse SVD
reconstructed_cA = np.dot(U, np.dot(np.diag(S), Vt))

# Extract the watermark
extracted_watermark = (reconstructed_cA - cA) / alpha

# Evaluation - Compare the extracted watermark with the original one
# For example, you can use metrics like correlation coefficient or Mean Squared Error (MSE)

# Display results
cv2.imshow('Original Image', host_img)
cv2.imshow('Watermarked Image', watermarked_img.astype(np.uint8))
cv2.imshow('Extracted Watermark', (extracted_watermark * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

