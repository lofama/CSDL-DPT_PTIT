import numpy as np
# Load file
data_file_hsv = np.load("HSV.npy", allow_pickle=True)
data_file_hog = np.load("HOG.npy", allow_pickle=True)
data_file_rgb = np.load("RGB.npy", allow_pickle=True)
y = np.load("y.npy",allow_pickle=True)
# Print số lượng mẫu trong data_file_hog
num_samples = len(data_file_hog[:, 1])
print("Số lượng mẫu trong data_file_hog:", num_samples)

array_concat_hog_hsv_rgb = []
# Tạo mảng concat_hog_hsv và concat_hog_rgb
for i in range(num_samples):
        concat_in_value1 = np.concatenate((data_file_hsv[i], data_file_hog[i]))
        concat_in_value = np.concatenate((concat_in_value1, data_file_rgb[i]))
        array_concat_hog_hsv_rgb.append( concat_in_value)
print(len(array_concat_hog_hsv_rgb[0]))

# Lưu mảng concat_hog_hsv và concat_hog_rgb vào các file npy
np.save("concat_hog_hsv_rgb.npy", array_concat_hog_hsv_rgb)

# combined_feature_vectors_hsv_hog = np.load("concat_hog_hsv.npy")
# print(len(combined_feature_vectors_hsv_hog[0]))
# print(combined_feature_vectors_hsv_hog[0])