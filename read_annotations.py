import json,os
from PIL import Image, ImageDraw
import cv2
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
# 读取标注数据
DATANO=64
script_directory = os.path.dirname(os.path.abspath(__file__))

json_path=os.path.join(script_directory,r"6th_integrated_circuit_eda_elite_challenge_question10_dataset\all_images"+f"\\{DATANO}.json")
image_path=os.path.join(script_directory,r"6th_integrated_circuit_eda_elite_challenge_question10_dataset\all_images"+f"\\{DATANO}.png")
with open(json_path, "r") as f:
    annotation_data = json.load(f)

image_width = annotation_data["imageWidth"]
image_height = annotation_data["imageHeight"]

# 打开图片
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

# 解析标注框
shapes = annotation_data["shapes"]

for shape in shapes:
    label = shape["label"]  # 获取类别标签
    points = shape["points"]  # 获取标注框的坐标点
    x1, y1 = points[0]
    x2, y2 = points[1]

    # 绘制矩形框
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # 绘制标签
    draw.text((x1, y1 - 10), label, fill="red")

# 显示带标注的图片
image.show()

# 保存结果
# output_path = f"annotated_image_{DATANO}.png"
# image.save(output_path)
# print(f"Annotated image saved to {output_path}")



