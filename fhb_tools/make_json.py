import os
import json


def find_jpg_files(path):
    # 初始化一个空列表来存储所有的jpg文件路径
    jpg_files = []

    # 使用os.walk()函数遍历指定路径下的所有文件和子目录
    for root, dirs, files in os.walk(path):
        for file in files:
            # 如果文件的扩展名是.jpg，就将它的绝对路径添加到列表中
            if file.endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))

    return jpg_files


def write_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


# 使用你想要搜索的路径替换 'your_path'
jpg_files = find_jpg_files('/home/lwt/work/mmdetection/data/MAR20/JPEGImages')

# 将结果写入JSON文件
write_to_json(jpg_files, 'jpg_files.json')
