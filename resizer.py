import cv2
import os

dir_path = os.getcwd()
file_extension = '.jpg'
resize_value = 0.35
directory_for_saving = 'resized_images'
chars_number_to_remove = 9
for filename in os.listdir(dir_path):
    if filename.endswith(file_extension):
        os.rename(filename, filename[chars_number_to_remove:])
for filename in os.listdir(dir_path):
    if filename.endswith(file_extension):
        image = cv2.imread(filename)
        resized_image = cv2.resize(image, None, fx=resize_value, fy=resize_value, interpolation=cv2.INTER_AREA)
        cv2.imwrite(directory_for_saving + '\\resized_' + filename, resized_image)
