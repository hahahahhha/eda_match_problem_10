import json
from analyze import LABEL_DICT
import os
import shutil
import random

PIC_NUM=2269
label_to_class_id={}

script_directory = os.path.dirname(os.path.abspath(__file__))

LABELS_DIR=os.path.join(script_directory,r'6th_integrated_circuit_eda_elite_challenge_question10_dataset\all_images')       # 原始标签文件夹路径
IMAGES_DIR=os.path.join(script_directory,r'6th_integrated_circuit_eda_elite_challenge_question10_dataset\all_images')      # 原始图片文件夹路径
OUTPUT_DIR=os.path.join(script_directory,r'datasets\eda')               # 输出数据集的根目录

for idx,key in enumerate(LABEL_DICT):
    label_to_class_id[key]=idx


def convert_to_yolo_format(json_path, output_txt_path):
    # 读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    yolo_annotations = []

    # 遍历每个形状
    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]

        # 获取边界框的坐标
        x_min = points[0][0]
        y_min = points[0][1]
        x_max = points[1][0]
        y_max = points[1][1]

        # 转换为 YOLO 格式
        center_x = (x_min + x_max) / 2 / image_width
        center_y = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        class_id = label_to_class_id.get(label)
        if class_id is None:
            print(f"Warning: Unknown label '{label}', skipping...")
            continue

        # 格式化为字符串
        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    # 写入 YOLO 格式文件
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_annotations))

    print(f"YOLO annotations saved to {output_txt_path}")

def split_dataset(labels_dir, images_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 创建目标目录结构
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        os.makedirs(os.path.join(output_dir, 'labels', subset), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', subset), exist_ok=True)
    
    # 获取所有标签文件和对应的图片文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
    image_files = [f.replace('.json', '.png') for f in label_files]

    # 确保每个标签文件都有对应的图片文件
    for label, image in zip(label_files, image_files):
        if not os.path.exists(os.path.join(images_dir, image)):
            print(f"Warning: Image file {image} not found for label {label}, skipping...")
            continue
    
    # 打乱文件顺序
    paired_files = list(zip(label_files, image_files))
    random.shuffle(paired_files)

    # 分割数据集
    total_files = len(paired_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    train_files = paired_files[:train_count]
    val_files = paired_files[train_count:train_count + val_count]
    test_files = paired_files[train_count + val_count:]

    # 复制文件到对应的文件夹
    def copy_files(subset_files, subset_name):
        for label, image in subset_files:
            convert_to_yolo_format(os.path.join(labels_dir, label),os.path.join(output_dir, 'labels', subset_name, label.replace('.json','.txt')))

            shutil.copy(os.path.join(images_dir, image), os.path.join(output_dir, 'images', subset_name, image))

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print("Dataset split completed!")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# 使用示例
if __name__ == '__main__':
    split_dataset(
        labels_dir=LABELS_DIR,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR
    )

