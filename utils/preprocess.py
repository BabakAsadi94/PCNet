import cv2
import numpy as np
import os

# Define the input filename
input_filename = 'MMM.jpg'

# Load the image
image = cv2.imread(input_filename)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"The image '{input_filename}' was not found.")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to segment the black part
# Pixels darker than 50 become white (foreground), others become black (background)
_, thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

# Perform morphological operations to clean up noise (optional)
kernel = np.ones((5, 5), np.uint8)
morph_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

# Find contours in the thresholded image
contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask to show the black part without filtering (including small black regions)
mask_with_small_black = np.zeros_like(gray_image)
cv2.drawContours(mask_with_small_black, contours, -1, 255, thickness=cv2.FILLED)

# Extract black part (with small black regions still present)
# Set background to black (0) instead of white (255)
extracted_black_part_with_small_blacks = np.zeros_like(image)
extracted_black_part_with_small_blacks[mask_with_small_black == 255] = image[mask_with_small_black == 255]

# Create a mask to filter out small black regions based on the distance from the center
mask = np.zeros_like(gray_image)
center_x, center_y = gray_image.shape[1] // 2, gray_image.shape[0] // 2
radius_threshold = 40  # Adjust this value as needed

# Filter contours based on their distance from the center
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        distance_from_center = np.sqrt((cX - center_x) ** 2 + (cY - center_y) ** 2)
        if distance_from_center < radius_threshold:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

# Create an empty black image for the final result
extracted_black_part = np.zeros_like(image)  # Start with black background
extracted_black_part[mask == 255] = image[mask == 255]

# Check if any contours were found
if not contours:
    raise ValueError("No contours found in the image. Please check the thresholding and image content.")

# Find the largest contour (which corresponds to the black circle)
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding box of the largest contour (the black circle)
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the black circle part
cropped_black_circle = extracted_black_part[y:y+h, x:x+w]

# Resize the cropped black circle to 224x224 using Lanczos interpolation
resized_black_circle = cv2.resize(cropped_black_circle, (224, 224), interpolation=cv2.INTER_LANCZOS4)

# Generate the output filename by replacing the input file's extension with .png
base_name, _ = os.path.splitext(input_filename)
output_filename = f"{base_name}.png"

# Save the final resized image as PNG
cv2.imwrite(output_filename, resized_black_circle)

print(f"Final resized image saved as '{output_filename}'.")
