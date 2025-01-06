import cv2
import numpy as np

src = cv2.imread('./parse_preds_mini_cropped_v4/PA000005_vessel_02570_right.png', 0)

# binary_map = (src > 0).astype(np.uint8)
# connectivity = 4 # or whatever you prefer
#
# output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
#
# print(output)


_, thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
connectivity = 4  # You need to choose 4 or 8 for connectivity type
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

print(num_labels)
print(labels)
print(stats)
print(centroids)
