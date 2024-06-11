import os

def rename_images(directory):
    # Lấy danh sách các file trong thư mục
    files = os.listdir(directory)
    
    # Lọc ra các file ảnh (có thể chỉnh sửa phần điều kiện nếu cần)
    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # Đếm số lượng file ảnh
    num_images = len(image_files)
    
    # Đổi tên các file
    for i, old_name in enumerate(image_files):
        # Tạo tên mới cho file ảnh
        new_name = f"{i+1}.jpg"  # Đổi phần mở rộng file tùy theo định dạng ảnh
        
        # Tạo đường dẫn đầy đủ cho file cũ và mới
        old_path = os.path.join(directory, old_name)
        
        # Kiểm tra nếu file mới đã tồn tại, thì thêm số vào tên file mới
        new_path = os.path.join(directory, new_name)
        while os.path.exists(new_path):
            i +=1
            new_name = f"{i+1}.jpg"
            new_path = os.path.join(directory, new_name)
        
        # Đổi tên file
        os.rename(old_path, new_path)
        print(f"Đã đổi tên '{old_name}' thành '{new_name}'")

# Thay đường dẫn thư mục của bạn vào đây
directory_path = "./new"

# Gọi hàm để đổi tên các file ảnh trong thư mục
rename_images(directory_path)
