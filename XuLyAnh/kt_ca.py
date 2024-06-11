import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def process_image(image_path, output_dir):
    # Đọc ảnh
    image = cv2.imread(image_path)

    # Kiểm tra xem ảnh có được đọc thành công không
    if image is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    # Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sử dụng ngưỡng để tách đối tượng khỏi nền
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Tìm các đường viền của đối tượng
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Giả sử đối tượng lớn nhất là con cá, lấy đường viền lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)

    # Tính toán hộp bao quanh đối tượng
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Cắt đối tượng từ ảnh gốc
    object_roi = image[y:y+h, x:x+w]

    # Tính diện tích của ảnh gốc và đối tượng
    image_area = image.shape[0] * image.shape[1]
    object_area = w * h

   # Tính tỷ lệ để đối tượng chiếm khoảng 50% diện tích ảnh
    target_area_ratio = 0.5
    object_area_sqrt = np.sqrt(object_area)
    image_area_sqrt = np.sqrt(image_area)
    scale_factor = np.sqrt(target_area_ratio * image_area) / object_area_sqrt


    # Thay đổi kích thước đối tượng
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)



    # Điều chỉnh kích thước nếu đối tượng vượt quá kích thước của ảnh gốc
    if new_w > image.shape[1]:
        new_w = image.shape[1]
    if new_h > image.shape[0]:
        new_h = image.shape[0]

    resized_object = cv2.resize(object_roi, (new_w, new_h))

    # Tạo ảnh mới với kích thước ban đầu và nền màu đen
    output_image = np.zeros_like(image)

    # Tính toán vị trí để chèn đối tượng vào giữa ảnh
    center_x = (image.shape[1] - new_w) // 2
    center_y = (image.shape[0] - new_h) // 2

    # Chèn đối tượng đã thay đổi kích thước vào ảnh mới
    output_image[center_y:center_y+new_h, center_x:center_x+new_w] = resized_object

    # Lưu ảnh đã xử lý
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, output_image)

    # Kiểm tra và hiển thị diện tích đối tượng sau khi thay đổi kích thước
    gray_output = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    _, threshold_output = cv2.threshold(gray_output, 1, 255, cv2.THRESH_BINARY)
    contours_output, _ = cv2.findContours(threshold_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour_output = max(contours_output, key=cv2.contourArea)
    object_area_output = cv2.contourArea(largest_contour_output)
    object_percentage = (object_area_output / image_area) * 100
    print(f"Diện tích đối tượng chiếm: {object_percentage:.2f}% diện tích ảnh")

    # Kiểm tra nếu object_percentage nhỏ hơn 20 hoặc lớn hơn 30, thì không lưu ảnh và kết thúc hàm
    if object_percentage < 20 or object_percentage > 35:
        print("Đối tượng chiếm diện tích quá nhỏ hoặc quá lớn. Không lưu ảnh.")
        # Xóa ảnh
        os.remove(output_path)
        return
    
def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)
            print(f"Processed {filename}")

# Đường dẫn đến thư mục chứa ảnh đầu vào và thư mục lưu ảnh đầu ra
input_directory = 'test0'
output_directory = 'test'

process_directory(input_directory, output_directory)
