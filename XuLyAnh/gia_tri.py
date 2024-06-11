import cv2
import numpy as np
import os

# Hàm để phân tích màu sắc của ảnh và trả về giá trị màu HSV chính
def analyze_image_color(image_path):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)
    
    # Chuyển đổi sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Tính toán giá trị trung bình của từng kênh màu
    avg_hue = np.mean(hsv_image[:,:,0])
    avg_saturation = np.mean(hsv_image[:,:,1])
    avg_value = np.mean(hsv_image[:,:,2])
    
    return avg_hue, avg_saturation, avg_value

# Đường dẫn đến thư mục chứa ảnh
folder_path = "path/to/your/image/folder"

# Lặp qua từng tệp ảnh trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Tạo đường dẫn đầy đủ đến ảnh
        image_path = os.path.join(folder_path, filename)
        
        # Phân tích màu sắc của ảnh
        avg_hue, avg_saturation, avg_value = analyze_image_color(image_path)
        
        # In ra các giá trị màu HSV của ảnh
        print(f"Color analysis for {filename}:")
        print(f"Average Hue: {avg_hue}")
        print(f"Average Saturation: {avg_saturation}")
        print(f"Average Value: {avg_value}")
        print("---------------------------------------")

print("Color analysis complete.")
