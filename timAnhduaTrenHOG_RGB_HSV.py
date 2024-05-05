import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
#imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import cv2
from skimage import feature
from sklearn.metrics.pairwise import euclidean_distances

import HienThi49Anh
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
    if file_path:
        display_images(file_path)
def hienThi49():
    HienThi49Anh.display_images_in_grid()
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

def find_most_similar_images(file_path, target_image_index, has):
    top_n=3
    load_y = np.load("y.npy",allow_pickle=True)
    path_dataset= r"img_data"
    path_img = file_path
    name = path_img.split("/")[len(path_img.split("/"))-1]
    print(name)
    file_names = []
    file_names.append(name)
    fish_data = []
    img_size = (496, 496)
    count = 0
    for name in file_names:
        img_array = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, img_size)
    # fish_data.append([class_num,img_array])
        fish_data.append([img_array])  # Chuyển đổi mảng hình ảnh thành mảng numpy
        print(count, end=" ")
        count=count+1
    data_RGB =[]
  # Đọc ảnh và chuyển đổi sang không gian màu HSV
    img = fish_data[0][0]
    bins = [8, 8, 8]
    ranges = [[0, 256], [0, 256], [0, 256]]
#   img_hsv=covert_image_rgb_to_hsv(img)
    hist_my = my_calcHist(img, [0, 1, 2], bins, ranges)
    embedding = hist_my.flatten()
    embedding[0]=0
    data_RGB.append(embedding)
    print("Data RGB:",data_RGB)
    data_HSV=[]
    bins = [8,12,3]
    ranges = [[0, 180], [0, 256], [0, 256]]
    img_hsv=covert_image_rgb_to_hsv(img)
    hist_my = my_calcHist(img_hsv, [0, 1, 2], bins, ranges)
  # print(hist_my.shape)
    embedding = hist_my.flatten()
    embedding[0]=0
    data_HSV.append(embedding)
    print("Data HSV:",data_HSV)
    data_hog=[]
    img_gray=convert_image_rgb_to_gray(fish_data[0][0])
    embedding=hog_feature(img_gray)
    embedding = embedding.flatten()
  # embedding[0]=0
    data_hog.append(embedding)
    print("Data HOG:",data_hog)
    array_concat_hsv_hog_rgb = []
    num_samples = len(fish_data)
    print("len(fish_data):",len(fish_data))
    for i in range(num_samples):
        concat_in_value = np.concatenate((data_HSV[i], data_hog[i]))
        concat_in_value1 = np.concatenate((concat_in_value,data_RGB[i]))
        array_concat_hsv_hog_rgb.append(concat_in_value1)
        # np.save("concat_hog_hsv_newimg.npy", array_concat_hog_hsv)
        # np.save("concat_hog_rgb_newimg.npy", array_concat_hog_rgb)
    print(len(array_concat_hsv_hog_rgb[0]))
    # Kết hợp feature vectors từ HSV và HOG
    combined_feature_vectors_hsv_hog = np.load("concat_hog_hsv_rgb.npy")
    if has:
        combined_feature_vectors_hsv_hog = np.concatenate((combined_feature_vectors_hsv_hog, array_concat_hsv_hog_rgb), axis=0)  # Nối các mảng

    print("target_image_index: ",target_image_index)
    # Tính toán khoảng cách Euclid giữa các feature vectors
    distances_hsv_hog = euclidean_distances(combined_feature_vectors_hsv_hog, combined_feature_vectors_hsv_hog)
    # Lấy chỉ số của ảnh mục tiêu
    target_image_distances_hsv_hog = distances_hsv_hog[target_image_index]

    # Sắp xếp các ảnh theo khoảng cách tới ảnh mục tiêu và lấy top_n ảnh gần nhất
    closest_images_indices_hsv_hog = np.argsort(target_image_distances_hsv_hog)[1:top_n+1]
    listImg = []
    # In ra thông tin về top_n ảnh gần nhất và phần trăm giống
    print("Top", top_n, "ảnh giống nhất với ảnh", target_image_index, "sử dụng HSV và HOG:")
    for i, index in enumerate(closest_images_indices_hsv_hog):
        similarity_percent = (1 - target_image_distances_hsv_hog[index] / np.max(distances_hsv_hog)) * 100
        print("Ảnh", i+1, ":", index, "- Phần trăm giống:", similarity_percent)
        listImg.append([i,load_y[index],"{:.2f}".format(similarity_percent)])
    return listImg 
def display_images(file_path):
    # Load the image
    target_image_index = 135
    load_y = np.load("y.npy",allow_pickle=True)
    listIMG = []
    file_names = file_path.split("/")[len(file_path.split("/"))-1]
    print(load_y)
    print(file_names)
    if file_names in load_y:
        target_image_index = np.where(load_y == file_names)[0][0]
        listIMG = find_most_similar_images(file_path,target_image_index, False)
    else:
        load_y = np.append(load_y,[file_names])
        target_image_index = len(load_y) - 1
        listIMG = find_most_similar_images(file_path,target_image_index,True)
        print(listIMG)
    image1 = Image.open("img_data/"+listIMG[0][1])
    print(listIMG[1][1])
    image2 = Image.open("img_data/"+listIMG[1][1])
    image3 = Image.open("img_data/"+listIMG[2][1])
    image = Image.open(file_path)
    # Resize the image to fit one-fourth of the original size
    width, height = image1.size
    new_width = int(width / 4)
    new_height = int(height / 4)
    resized_image1 = image1.resize((new_width, new_height))
    resized_image2 = image2.resize((new_width, new_height))
    resized_image3 = image3.resize((new_width, new_height))
    resized_image4 = image.resize((new_width*2, new_height*2))
    
    # Create Tkinter image objects
    tk_image1 = ImageTk.PhotoImage(resized_image1)
    tk_image2 = ImageTk.PhotoImage(resized_image2)
    tk_image3 = ImageTk.PhotoImage(resized_image3)
    tk_image4 = ImageTk.PhotoImage(resized_image4)
    # Display images
    label1.config(image=tk_image1)
    label2.config(image=tk_image2)
    label3.config(image=tk_image3)
    label4.config(image=tk_image4)
    
    # Keep references to avoid garbage collection
    label1.image = tk_image1
    label2.image = tk_image2
    label3.image = tk_image3
    label4.image = tk_image4
    name1 = listIMG[0][1]+" Giống:"+str(listIMG[0][2])+"%"
    name2 = listIMG[1][1]+" Giống:"+str(listIMG[1][2])+"%"
    name3 = listIMG[2][1]+" Giống:"+str(listIMG[2][2])+"%"
    # Display image name
    image_name_label1.config(text=name1)
    image_name_label2.config(text=name2)
    image_name_label3.config(text=name3)
    image_name_label4.config(text=file_path)
# Create the main application window
root = tk.Tk()
root.title("Find Image")

# Create labels for displaying images
label1 = tk.Label(root, text="")
label1.grid(row=0, column=0)

label2 = tk.Label(root)
label2.grid(row=0, column=1)

label3 = tk.Label(root)
label3.grid(row=0, column=2)

# Create label for displaying image name
image_name_label1 = tk.Label(root, text="")
image_name_label1.grid(row=1, column=0)
image_name_label2 = tk.Label(root, text="")
image_name_label2.grid(row=1, column=1)
image_name_label3 = tk.Label(root, text="")
image_name_label3.grid(row=1, column=2)
# Create a canvas to display images
canvas = tk.Canvas(root)
canvas.grid(row=2, column=0, columnspan=3, sticky="nsew")
label4 = tk.Label(root)
label4.grid(row=2, column=1)
# Create and place button for opening image
image_name_label4 = tk.Label(root, text="")
image_name_label4.grid(row=2, column=0)
open_button = tk.Button(root, text="Find Image", command=open_image)
open_button.grid(row=3, column=1)
open_button = tk.Button(root, text="Show Image", command=hienThi49)
open_button.grid(row=3, column=2)

# Start the Tkinter event loop
root.mainloop()