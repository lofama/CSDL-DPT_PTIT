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
import re

path_dataset= r"img_data"
# # data clean là thư mục , trong ddoscos nhiều thư mục con
# dir = path_dataset
# datadir = path_dataset

# Lấy danh sách tên tất cả các tệp trong thư mục chứa ảnh
file_names = os.listdir(path_dataset)

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
    fish_data.append([name,img_array])  # Chuyển đổi mảng hình ảnh thành mảng numpy
    print(count, end=" ")
    count=count+1
print()
X = []
y = []
# Chia dữ liệu thành X (hình ảnh) và y (nhãn)
X = np.array([item[1] for item in fish_data])
y = np.array([item[0] for item in fish_data])
# np.save("X.npy", X)
# np.save("y.npy", y)
print(y)
"""
listname = []
for i in file_names:
    num = i.split(".")[0]
    listname.append(num)
# 
# X = np.array([item[0] for item in fish_data])
# y = np.array([item[0] for item in listname])
## print(X)
## print(y)
# Lưu dữ liệu vào hai tệp numpy riêng biệt
# np.save("X.npy", X)
# np.save("y.npy", y)
# np.save("fish_data.npy",fish_data)
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
# Trích xuất dữ liệu RGB

//
data_RGB =[]
for i in range(len(fish_data)):
  # Đọc ảnh và chuyển đổi sang không gian màu HSV
  img = fish_data[i][1]
  bins = [8, 8, 8]
  ranges = [[0, 256], [0, 256], [0, 256]]
#   img_hsv=covert_image_rgb_to_hsv(img)
  hist_my = my_calcHist(img, [0, 1, 2], bins, ranges)
  embedding = hist_my.flatten()
  embedding[0]=0
  data_RGB.append(embedding)
  print(i,end=' ')
print(data_RGB)
np.save("RGB.npy",data_RGB)

data_HSV=[]
for i in range(len(fish_data)) :
  # Đọc ảnh và chuyển đổi sang không gian màu HSV
  img = fish_data[i][1]
  bins = [8,12,3]
  ranges = [[0, 180], [0, 256], [0, 256]]
  img_hsv=covert_image_rgb_to_hsv(img)
  hist_my = my_calcHist(img_hsv, [0, 1, 2], bins, ranges)
  # print(hist_my.shape)
  embedding = hist_my.flatten()
  embedding[0]=0
  data_HSV.append(embedding)
  print(i,end=' ')
np.save("HSV.npy", data_HSV)

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
  img_gray=convert_image_rgb_to_gray(fish_data[i][1])
  embedding=hog_feature(img_gray)
  embedding = embedding.flatten()
  # embedding[0]=0
  data_hog.append(embedding)
  print(i,end=' ')
np.save("HOG.npy",data_hog)
"""
#Load file
data_file_hsv=np.load("HSV.npy",allow_pickle=True)
data_file_hog=np.load("HOG.npy",allow_pickle=True)
data_file_rgb=np.load("RGB.npy",allow_pickle=True)
print(len(data_file_hog[:,1]))
array_concat_hog_hsv=[]
for i in range(len(data_file_hog)):
  if data_file_hsv[:,1][i].ndim > 0 and data_file_hog[:,1][i].ndim > 0:
    concat_in_value=np.concatenate((data_file_hsv[:,1][i], data_file_hog[:,1][i]))
    array_concat_hog_hsv.append([data_file_hog[:,0][i], concat_in_value])
  # concat_in_value=np.concatenate((data_file_hsv[:,1][i] ,data_file_hog[:,1][i]))
  # array_concat_hog_hsv.append([data_file_hog[:,0][i],concat_in_value])
np.save("concat_hog_hsv.npy", array_concat_hog_hsv)
array_concat_hog_rgb=[]
for i in range(len(data_file_hog)):
  if data_file_rgb[:,1][i].ndim > 0 and data_file_hog[:,1][i].ndim > 0:
    concat_in_value=np.concatenate((data_file_rgb[:,1][i] ,data_file_hog[:,1][i]))
    array_concat_hog_rgb.append([data_file_hog[:,0][i],concat_in_value])
np.save("concat_hog_rgb.npy",array_concat_hog_rgb)