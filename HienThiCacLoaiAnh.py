#imports
import pandas as pd
import seaborn as sns
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import cv2
from skimage import exposure
from skimage import feature
from tqdm import tqdm
import matplotlib.pyplot as plt
path_dataset= r"img_data"
# # data clean là thư mục , trong ddoscos nhiều thư mục con
# dir = path_dataset
# datadir = path_dataset

# Lấy danh sách tên tất cả các tệp trong thư mục chứa ảnh
# file_names = os.listdir(path_dataset)
file_names = ["100.jpg"]
# Lấy tổng số tệp ảnh trong thư mục
# total_images = len(file_names)
# print("Tổng số ảnh:", total_images)
# print("Tên:",file_names)
fish_data = []
img_size = (496, 496)
count = 0
for name in file_names:
    img_array = cv2.imread(os.path.join(path_dataset, name), cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, img_size)
    # fish_data.append([class_num,img_array])
    fish_data.append([img_array])  # Chuyển đổi mảng hình ảnh thành mảng numpy
    print(count, end=" ")
    count=count+1
print()

def rgb_to_hsv(pixel):
    r , g, b = pixel 
    r , g ,b = b / 255.0, g / 255.0, r / 255.0
    
    v = max(r,g,b)
    delta = v - min(r,g,b)
    
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / v
        if r == v:
            h = (g - b) / delta
        elif g == v:
            h = 2 + (b - r) / delta
        else:
            h = 4 + (r - g) / delta
        h = (h / 6) % 1.0
        
    return [int(h*180), int(s*255), int(v*255)]

def covert_image_rgb_to_hsv(img):
  hsv_image=[]
  for i in img:
    hsv_image2=[]
    for j in i:
      new_color=rgb_to_hsv(j)
      hsv_image2.append((new_color))
    hsv_image.append(hsv_image2)
  hsv_image=np.array(hsv_image)
  return hsv_image
def my_calcHist(image, channels, histSize, ranges):
    # Khởi tạo histogram với tất cả giá trị bằng 0
    hist = np.zeros(histSize, dtype=np.int64)
    # Lặp qua tất cả các pixel trong ảnh
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Lấy giá trị của kênh màu được chỉ định
            bin_vals = [image[i, j, c] for c in channels]
            # Tính chỉ số của bin
            bin_idxs = [(bin_vals[c] - ranges[c][0]) * histSize[c] // (ranges[c][1] - ranges[c][0]) for c in range(len(channels))]
            # Tăng giá trị của bin tương ứng lên 1
            hist[tuple(bin_idxs)] += 1
    return hist
data_RGB =[]
for i in range(len(fish_data)):
  # Đọc ảnh và chuyển đổi sang không gian màu HSV
  img = fish_data[i][0]
  bins = [8, 8, 8]
  ranges = [[0, 256], [0, 256], [0, 256]]
#   img_hsv=covert_image_rgb_to_hsv(img)
  hist_my = my_calcHist(img, [0, 1, 2], bins, ranges)
  embedding = hist_my.flatten()
  embedding[0]=0
  data_RGB.append(embedding)
  print(i,end=' ')
print("Data RGB:",data_RGB)
data_HSV=[]
for i in range(len(fish_data)) :
  # Đọc ảnh và chuyển đổi sang không gian màu HSV
  img = fish_data[i][0]
  bins = [8,12,3]
  ranges = [[0, 180], [0, 256], [0, 256]]
  img_hsv=covert_image_rgb_to_hsv(img)
  hist_my = my_calcHist(img_hsv, [0, 1, 2], bins, ranges)
  # print(hist_my.shape)
  embedding = hist_my.flatten()
  embedding[0]=0
  data_HSV.append(embedding)
  print(i,end=' ')
print("Data HSV:",data_HSV)

# Trich xuat dac trug hinh dang
def convert_image_rgb_to_gray(img_rgb, resize="no"):
  h, w, _ = img_rgb.shape
  # Create a new grayscale image with the same height and width as the RGB image
  img_gray = np.zeros((h, w), dtype=np.uint32)

  # Convert each pixel from RGB to grayscale using the formula Y = 0.299R + 0.587G + 0.114B
  for i in range(h):
      for j in range(w):
          r, g, b = img_rgb[i, j]
          gray_value = int(0.299*r + 0.587*g + 0.114*b)
          img_gray[i, j] = gray_value
  # print(gray_image.shape())
  if resize!="no":
     img_gray = cv2.resize(src=img_gray, dsize=(496, 496))
  return np.array(img_gray)
def hog_feature(gray_img):# default gray_image
  # 1. Khai báo các tham số
  (hog_feats, hogImage) = feature.hog(gray_img, orientations=9, pixels_per_cell=(8 , 8),
    cells_per_block=(2,2), transform_sqrt=True, block_norm="L2",
    visualize=True)
  return hog_feats
# Trich xuat HOG
data_hog=[]
for i in range(len(fish_data)) :
  # Đọc ảnh và chuyển đổi sang không gian màu HSV

  # img_hsv=covert_image_rgb_to_hsv(img)
  # hist_my = my_calcHist(img_hsv, [0, 1, 2], bins, ranges)
  # print(hist_my.shape)
  img_gray=convert_image_rgb_to_gray(fish_data[i][0])
  embedding=hog_feature(img_gray)
  embedding = embedding.flatten()
  # embedding[0]=0
  data_hog.append(embedding)
  print(i,end=' ')
print("Data HOG:",data_hog)

array_concat_hog_hsv = []
array_concat_hog_rgb = []
num_samples = len(fish_data)
for i in range(num_samples):
    concat_in_value = np.concatenate((data_HSV[i], data_hog[i]))
    array_concat_hog_hsv.append([data_hog[i], concat_in_value])

    concat_in_value = np.concatenate((data_RGB[i], data_hog[i]))
    array_concat_hog_rgb.append([data_hog[i], concat_in_value])
print(len(array_concat_hog_hsv))



# Function to convert RGB image to HSV
def convert_rgb_to_hsv(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return img_hsv

# Function to display image
def show_image(img, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load RGB image
img_rgb = fish_data[0][0]

# Convert RGB image to HSV
img_hsv = convert_rgb_to_hsv(img_rgb)

# Display RGB image
show_image(img_rgb, 'RGB Image')

# Display HSV image
show_image(img_hsv, 'HSV Image')

# Calculate and display HOG features
gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
hog_feats, hog_image = feature.hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                   visualize=True)
show_image(hog_image, 'HOG Image')
show_image(gray_img, 'GRAY Image')