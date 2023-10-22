import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_dataset_mean_and_variance(folder_path):
    pixel_sum = 0
    pixel_squared_sum = 0
    total_pixels = 0

    # 遍历文件夹下的所有文件
    for filename in tqdm(os.listdir(folder_path)):
        # 检查文件是否为.jpg格式
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)

            # 打开图像并转换为numpy数组
            image = Image.open(image_path)
            image_array = np.array(image)

            total_pixels += image_array.size
            pixel_sum += np.sum(image_array)
            pixel_squared_sum += np.sum(np.square(image_array - np.mean(image_array)))

    mean = pixel_sum / total_pixels
    variance = pixel_squared_sum / total_pixels

    return mean, variance

def main():
    folder_path = '/home/lwt/mmdetection/data/SAR-AIRcraft-1.0/JPEGImages'  # 替换为你的图像文件夹路径

    mean, variance = calculate_dataset_mean_and_variance(folder_path)
    print(f'Dataset Mean: {mean}, Dataset Variance: {variance}')

if __name__ == '__main__':
    main()