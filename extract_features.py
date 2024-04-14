import cv2
import pywt
# from scipy.linalg import svd
import numpy as np

def extract_features(watermarked_image):
  # Assuming the image is loaded in grayscale format
  # Step 1: Apply filter and extract HL3 sub-bands (replace with your specific filtering method)
  # This part might involve libraries like OpenCV (cv2) for filtering
  HL3_coefficients = filter_image(watermarked_image, filter_type="HL3")
  
  # Step 2: Divide into blocks
  block_size = (8, 8) # Adjust block size as needed
  blocks = image_to_blocks(HL3_coefficients, block_size)
  
  # Step 3: Perform SVD on each block
  features = []
  for block in blocks:
    singular_values, _, _ = np.linalg.svd(block)
    # Step 4: Calculate statistical parameters from singular values
    # Here, we calculate mean and standard deviation, but you can choose other statistics
    mean_singular_value = np.mean(singular_values)
    std_singular_value = np.std(singular_values)
    features.append([mean_singular_value, std_singular_value])
  
  return features

def filter_image(image, filter_type="HL3"):
    """
    This function applies a basic wavelet filter to extract HL3 sub-bands (replace for specific needs)

    Args:
        image: Grayscale image as a numpy array
        filter_type: String specifying filter type (assumed to be HL3 here)

    Returns:
        Filtered image containing HL3 sub-bands (replace for specific implementation)
    """
    # Pad the image if its dimensions are not divisible by 2
    height, width = image.shape
    if height % 2 != 0:
        pad_height = 2 - (height % 2)
    else:
        pad_height = 0
    if width % 2 != 0:
        pad_width = 2 - (width % 2)
    else:
        pad_width = 0
    image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant')

    # Apply Haar wavelet transform for HL3 sub-bands (adjust as needed)
    coeffs = pywt.dwt2(image, 'haar')
    HL3_subband, _ = coeffs  # Select the HL3 sub-band
    return HL3_subband


def image_to_blocks(image, block_size=(8, 8)):
  """
  This function divides an image into non-overlapping blocks

  Args:
      image: Grayscale image as a numpy array
      block_size: A tuple representing the size of each block (e.g., (8, 8))

  Returns:
      A list of image blocks
  """
  # Get image height and width
  image_height, image_width = image.shape

  # Get number of blocks horizontally and vertically
  num_blocks_h = int(image_width / block_size[0])
  num_blocks_v = int(image_height / block_size[1])

  # Initialize an empty list to store blocks
  blocks = []

  # Iterate through rows and columns to extract blocks
  for y in range(num_blocks_v):
    for x in range(num_blocks_h):
      # Define starting and ending indices for current block
      start_row = y * block_size[1]
      end_row = start_row + block_size[1]
      start_col = x * block_size[0]
      end_col = start_col + block_size[0]

      # Extract the current block and append it to the list
      block = image[start_row:end_row, start_col:end_col]
      blocks.append(block)

  return blocks

# Load your watermarked image as grayscale (replace with your image path)
image = cv2.imread("watermarked_image.png", cv2.IMREAD_GRAYSCALE)

# Extract features
features = extract_features(image)

# Print some sample features (these won't reveal the watermark directly)
print("Sample features:")
for feature in features[:5]:
  print(feature)
