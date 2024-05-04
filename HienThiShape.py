import cv2
import numpy as np
from skimage import feature

def process_image(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path)

    # Chuyển đổi ảnh thành kích thước 496x496
    resized_img = cv2.resize(img, (496, 496))

    # Trích xuất tính năng HOG
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    hog_features = feature.hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

    # Trích xuất tính năng HSV
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

    # Trích xuất tính năng RGB
    rgb_features = resized_img.flatten()

    return hog_features, hsv_img, rgb_features

# Thí dụ sử dụng hàm process_image với một đường dẫn ảnh cụ thể
image_path = "5.jpg"
hog_features, hsv_img, rgb_features = process_image(image_path)

print("HOG features shape:", hog_features.shape)
print("HSV image shape:", hsv_img.shape)
print("RGB features shape:", rgb_features.shape)
