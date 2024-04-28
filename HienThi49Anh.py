import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def display_images_in_grid(path_dataset=r"img_data", grid_size=(7, 7), figsize=(50, 50)):
    # Lấy danh sách tên tất cả các tệp ảnh trong thư mục
    file_names = os.listdir(path_dataset)
    total_images = len(file_names)
    print("Tổng số ảnh:", total_images)

    W, H = grid_size
    fig, axes = plt.subplots(W, H, figsize=figsize)

    axes = axes.ravel()
    for i in np.arange(0, W * H):
        # Chọn ngẫu nhiên một hình ảnh từ danh sách các tệp tin trong thư mục
        image = random.choice(file_names)
        image_path = os.path.join(path_dataset, image)
        # Đọc hình ảnh từ đường dẫn đầy đủ của hình ảnh đã chọn
        img = cv2.imread(image_path)
        # Hiển thị hình ảnh lên ô hình ảnh thứ i trong lưới
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(image, fontsize=10)
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

# Đường dẫn đến thư mục chứa ảnh
path_dataset = r"img_data"
# # Kích thước lưới và kích thước hình ảnh trong lưới
# grid_size = (7, 7)
# figsize = (50, 50)

# # Hiển thị hình ảnh trong lưới
# display_images_in_grid(path_dataset, grid_size, figsize)
