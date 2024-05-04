import numpy as np
# Load file
data_file_hsv = np.load("HSV.npy", allow_pickle=True)
data_file_hog = np.load("HOG.npy", allow_pickle=True)
data_file_rgb = np.load("RGB.npy", allow_pickle=True)
y = np.load("y.npy",allow_pickle=True)
# Print số lượng mẫu trong data_file_hog
num_samples = len(data_file_hog[:, 1])
print("Số lượng mẫu trong data_file_hog:", num_samples)

array_concat_hog_hsv = []
array_concat_hog_rgb = []

# Tạo mảng concat_hog_hsv và concat_hog_rgb
for i in range(num_samples):
        concat_in_value = np.concatenate((data_file_hsv[i], data_file_hog[i]))
        array_concat_hog_hsv.append( concat_in_value)

    # if data_file_rgb[:, 0][i].ndim > 0 and data_file_hog[:, 0][i].ndim > 0:
        concat_in_value = np.concatenate((data_file_rgb[i], data_file_hog[i]))
        array_concat_hog_rgb.append(concat_in_value)
print(len(array_concat_hog_hsv))
print(array_concat_hog_hsv[100])
# Lưu mảng concat_hog_hsv và concat_hog_rgb vào các file npy
np.save("concat_hog_hsv.npy", array_concat_hog_hsv)
np.save("concat_hog_rgb.npy", array_concat_hog_rgb)