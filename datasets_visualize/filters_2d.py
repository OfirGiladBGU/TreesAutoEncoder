import os
import cv2
import matplotlib.pyplot as plt
from configs.configs_parser import DATA_PATH

# Load the image
image_path = os.path.join(DATA_PATH, r"PipeForge3DPCD\labels_2d\01_back.png")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Bilateral Filter
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

# Apply Median Filter
k = 5 # 3
median_filter = cv2.medianBlur(image, k)

# Display the results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(gaussian_blur, cmap='gray')
axes[1].set_title('Gaussian Blur')
axes[1].axis('off')

axes[2].imshow(bilateral_filter, cmap='gray')
axes[2].set_title('Bilateral Filter')
axes[2].axis('off')

axes[3].imshow(median_filter, cmap='gray')
axes[3].set_title('Median Filter')
axes[3].axis('off')

plt.show()
