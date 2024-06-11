import cv2
import numpy as np
import os

# Đường dẫn tới thư mục chứa các ảnh
image_folder = "./test_black/"
# Đường dẫn tới thư mục để lưu các ảnh đã xử lý
output_folder = "./test/"

# Kích thước mới cho các ảnh (ở đây là 500x500)
new_width = 500
new_height = 500

# Lặp qua từng tệp ảnh trong thư mục
for filename in os.listdir(image_folder):
    # Đọc ảnh
    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)

    # Chuyển đổi ảnh sang ảnh đen trắng
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tìm contours của vật thể trong ảnh
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo một mask để chỉ chứa con cá (màu trắng) và nền (màu đen)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Áp dụng mask để loại bỏ phần nền từ ảnh gốc
    result_img = np.zeros_like(img)
    result_img[mask == 255] = img[mask == 255]

    # Resize ảnh mới về kích thước mong muốn
    result_img = cv2.resize(result_img, (new_width, new_height))

    # Lưu ảnh đã xử lý vào thư mục đầu ra
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result_img)

    print(f"Đã xử lý: {filename}")

print("Hoàn thành xử lý ảnh!")
