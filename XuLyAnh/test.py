import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image_path = './imgdata/44.jpg'
image = cv2.imread(image_path)

# Chuyển đổi sang ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sử dụng ngưỡng để tách đối tượng khỏi nền
_, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Tìm các đường viền của đối tượng
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Giả sử đối tượng lớn nhất là con cá, lấy đường viền lớn nhất
largest_contour = max(contours, key=cv2.contourArea)

# Tính toán diện tích của đối tượng và diện tích của ảnh
object_area = cv2.contourArea(largest_contour)
image_area = image.shape[0] * image.shape[1]

# Tính tỷ lệ diện tích của đối tượng so với diện tích ảnh
object_percentage = (object_area / image_area) * 100

print(f"Diện tích đối tượng chiếm: {object_percentage:.2f}% diện tích ảnh")

# Hiển thị kết quả
cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Object Area: {object_percentage:.2f}%")
plt.show()
