import os
from rembg import remove
from PIL import Image

# Thư mục chứa ảnh gốc
input_dir = "./new/"

# Thư mục để lưu các ảnh đã xử lý
output_dir = "./test_black/"

# Kiểm tra xem thư mục output có tồn tại chưa, nếu không thì tạo mới
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Duyệt qua từng tệp trong thư mục input_dir
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Đường dẫn đầy đủ của ảnh đang xử lý
        image_path = os.path.join(input_dir, filename)
        
        # Mở ảnh
        img = Image.open(image_path)
        
        # Loại bỏ nền
        R = remove(img)
        
        # Chuyển đổi sang chế độ RGB
        R = R.convert("RGB")
        
        # Lưu ảnh đã xử lý vào thư mục output với tên giống như tên của ảnh gốc
        output_path = os.path.join(output_dir, filename)
        R.save(output_path)
