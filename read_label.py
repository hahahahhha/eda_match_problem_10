import json
from PIL import Image, ImageDraw
import cv2
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

# 读取标注数据
DATANO=38
json_path=r"C:\Users\13617\Desktop\mycode\eda_match\6th_integrated_circuit_eda_elite_challenge_question10_dataset\all_images"+f"\\{DATANO}.json"
image_path=r"C:\Users\13617\Desktop\mycode\eda_match\6th_integrated_circuit_eda_elite_challenge_question10_dataset\all_images"+f"\\{DATANO}.png"

with open(json_path, "r") as f:
    annotation_data = json.load(f)
# 获取图片尺寸
image_width = annotation_data["imageWidth"]
image_height = annotation_data["imageHeight"]


# # 遍历轮廓并过滤出有效连线
# valid_contours = []  # 存储有效的连线轮廓
# for contour in contours:
#     if cv2.contourArea(contour) > 50:  # 过滤掉面积过小的轮廓
#         # 计算轮廓的最小外接矩形
#         rect = cv2.boundingRect(contour)
#         x, y, w, h = rect
        
#         # 创建掩膜，添加 padding 用于检测连线与元件区域的关系
#         padding = 1
#         mask_with_padding = np.zeros_like(full_mask)
#         cv2.rectangle(mask_with_padding, (x - padding, y - padding), (x + w + padding, y + h + padding), 255, -1)
        
#         # 只保留与元件区域有交集的连线
#         intersection_mask = cv2.bitwise_and(mask_with_padding, full_mask)
        
#         if cv2.countNonZero(intersection_mask) > 0:  # 如果有交集
#             valid_contours.append(contour)

#             # # 获取交集的坐标，记录端口
#             # contour_ports = []
#             # for shape in annotation_data["shapes"]:  # 遍历标注数据
#             #     # 获取标注框的坐标
#             #     points = shape["points"]
#             #     x1, y1 = points[0]
#             #     x2, y2 = points[1]
#             #     # 强制转换为整数
#             #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             #     # 创建元件的掩膜
#             #     element_mask = np.zeros_like(full_mask)
#             #     cv2.rectangle(element_mask, (x1, y1), (x2, y2), 255, -1)

#             #     # 计算连线与每个元件的交集
#             #     element_intersection = cv2.bitwise_and(intersection_mask, element_mask)

#             #     if cv2.countNonZero(element_intersection) > 0:  # 如果有交集，记录这个元件
#             #         contour_ports.append(shape["label"])  # 使用 label 记录该连线与元件的交集位置（端口）

#             # # 存储连线的端口信息
#             # if contour_ports:
#             #     # contour_tuple = tuple(map(tuple, contour))  # 将轮廓的所有点转换为元组
#             #     # connection_ports[contour_tuple] = contour_ports  # 使用元组作为字典的键
# # 打印或保存交集信息
# # 膨胀操作合并边缘
# line_image = np.copy(image)  # 复制图像用于绘制
# print(valid_contours)
# dilated_edges = cv2.dilate(edges_without_components, None, iterations=1)  # 进行膨胀操作
# for contour in dilated_edges:
#     cv2.drawContours(line_image, [contour], -1, (0, 255, 0), 2)  # 绘制有效连线


# # print("连接端口信息：")
# # for contour, ports in connection_ports.items():
# #     print(f"连线 {contour} 与元件 {ports} 有交集")
# # 显示图像
# plt.figure(figsize=(12, 8))

outputdict={'ckt_netlist':[],
            'ckt_type':'LDO'
            }

unit_list=[]
def net_element(component_type:str=None, 
                port_connection:dict = None):
    return{'component_type': component_type,
            'port_connection': port_connection}

def labelname2output(componentname : str):
    label_dict = {
        'pmos': 'PMOS',
        'pmos-normal': 'PMOS',
        'pmos-cross': 'PMOS',
        'pmos-bulk': 'PMOS',
        'nmos': 'NMOS',
        'nmos-normal': 'NMOS',
        'nmos-cross': 'NMOS',
        'nmos-bulk': 'NMOS',
        'voltage': 'Voltage',
        'voltage-1': 'Voltage',
        'voltage-2': 'Voltage',
        'voltage-lines': 'Voltage',
        'current': 'Current',
        'bjt-npn': 'NPN',
        'bjt-npn-cross': 'NPN',
        'bjt-pnp': 'PNP',
        'bjt-pnp-cross': 'PNP',
        'diode': 'Diode',
        'diso-amp': 'Diso_amp',
        'siso-amp': 'Siso_amp',
        'dido-amp': 'Dido_amp',
        'capacitor': 'Cap',
        'gnd': 'GND',
        'vdd': 'VDD',
        'inductor': 'Ind',
        'resistor': 'Res',
        'resistor-1': 'Res',
        'resistor-2': 'Res',

        'cross-line-curved':'cross-line-curved',
        'port': 'port'
    }

    ouputname=label_dict.get(componentname , None)
    if  ouputname is None:
        print(f'labelname2output:unknown labelname of component: {componentname}')

    return ouputname

def defaultconnnection(componentname : str):
    port_dict = {
        'pmos': {'Drain':None, 'Source':None, 'Gate':None},
        'pmos-normal': {'Drain':None, 'Source':None, 'Gate':None},
        'pmos-cross': {'Drain':None, 'Source':None, 'Gate':None},
        'pmos-bulk': {'Drain':None, 'Source':None, 'Gate':None, 'Body':None},
        'nmos': {'Drain':None, 'Source':None, 'Gate':None},
        'nmos-normal': {'Drain':None, 'Source':None, 'Gate':None},
        'nmos-cross': {'Drain':None, 'Source':None, 'Gate':None},
        'nmos-bulk': {'Drain':None, 'Source':None, 'Gate':None, 'Body':None},
        'voltage': {'Positive':None, 'Negative':None},
        'voltage-1': {'Positive':None, 'Negative':None},
        'voltage-2': {'Positive':None, 'Negative':None},
        'voltage-lines': {'Positive':None, 'Negative':None},
        'current': {'Positive':None, 'Negative':None},
        'bjt-npn': {'Base':None, 'Emitter':None, 'Collect':None},
        'bjt-npn-cross': {'Base':None, 'Emitter':None, 'Collect':None},
        'bjt-pnp': {'Base':None, 'Emitter':None, 'Collect':None},
        'bjt-pnp-cross': {'Base':None, 'Emitter':None, 'Collect':None},
        'diode': {'In':None, 'Out':None},
        'diso-amp': {'InN':None, 'InP':None, 'Out':None},
        'siso-amp': {'In':None, 'Out':None},
        'dido-amp': {'InN':None, 'InP':None, 'OutN':None, 'OutP':None},
        'capacitor': {'Pos':None, 'Neg':None},
        'gnd': {'port':None},
        'vdd': {'port':None},
        'inductor': {'Pos':None, 'Neg':None},
        'resistor': {'Pos':None, 'Neg':None},
        'resistor-1': {'Pos':None, 'Neg':None},
        'resistor-2': {'Pos':None, 'Neg':None},

        'cross-line-curved':{'Up':None,'Down':None,'Left':None,'Right':None},
        'port': {'port':None}

    }
    ouputconnection=port_dict.get(componentname , None)
    if  ouputconnection is None:
        print(f'defaultconnnection:unknown labelname of component: {componentname}')

    return ouputconnection

def init_net_element(componentname : str):
    return net_element(labelname2output(componentname), defaultconnnection(componentname))

def init_unit_element(componentname:str):
    return{'component_type': labelname2output(componentname),
            'port_connection': defaultconnnection(componentname),
            'label_type':componentname}

image = cv2.imread(image_path)  # 读取彩色图
if image is None:
    raise FileNotFoundError(f"Image {image_path} not found!")

# 获取图像尺寸
height, width = image.shape[:2]

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊减少噪声
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), sigmaX=1)

# 使用Canny边缘检测
edges = cv2.Canny(blurred_image, 0, 150)

# 创建空白掩膜，初始化为全0
full_mask = np.zeros((height, width), dtype=np.uint8)

padding=1
full_mask_padding = np.zeros((height, width), dtype=np.uint8)

# 处理标注框，生成掩膜
for shape in annotation_data["shapes"]:

    unit_list.append(init_unit_element(shape['label']))

    points = shape["points"]
    x1, y1 = points[0]
    x2, y2 = points[1]

    # 创建一个PIL图像对象来绘制矩形框
    mask = Image.new('L', (width, height), 0)  # 创建黑色背景（全0掩膜）
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)  # 填充矩形区域为白色

    # 将PIL图像转回为NumPy数组
    component_mask = np.array(mask)

    full_mask = cv2.bitwise_or(full_mask, component_mask)

    mask_padding = Image.new('L', (width, height), 0)  # 创建黑色背景（全0掩膜）
    draw = ImageDraw.Draw(mask_padding)
    draw.rectangle([x1-padding, y1-padding, x2+padding, y2+padding], fill=255)  # 填充矩形区域为白色
    component_mask_padding = np.array(mask_padding)
    full_mask_padding = cv2.bitwise_or(full_mask_padding, component_mask_padding)

# 反转掩膜，得到框外区域
inverse_mask = cv2.bitwise_not(full_mask)

# inverse_mask_padding = cv2.bitwise_not(full_mask_padding)

# 使用掩膜和边缘图像，去除元件区域
edges_without_components = cv2.bitwise_and(edges, edges, mask=inverse_mask)

# 查找轮廓
contours_original, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_without_components, _ = cv2.findContours(edges_without_components, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个字典来记录从属关系，键是去除元件区域的轮廓，值是原始图像中相关的轮廓
contour_relations = {}
# 遍历去除元件区域后的轮廓，找到与之相关的原始图像中的轮廓

for contour_without in contours_without_components:
    # 创建一个掩膜来绘制当前的轮廓
    mask_without = np.zeros_like(edges, dtype=np.uint8)
    cv2.drawContours(mask_without, [contour_without], -1, 255, thickness=cv2.FILLED)
    
    # 将 contour_without 转换为元组（不可变类型），作为字典的键
    contour_without_tuple = tuple(map(tuple, contour_without.reshape(-1, 2)))
    
    # 遍历原始图像中的轮廓，判断哪些轮廓在当前轮廓的区域内
    for contour_original in contours_original:
        mask_original = np.zeros_like(edges, dtype=np.uint8)
        cv2.drawContours(mask_original, [contour_original], -1, 255, thickness=cv2.FILLED)

        # 计算交集
        intersection = cv2.bitwise_and(mask_without, mask_original)
        
        # 如果交集区域非零，说明原始轮廓与去除元件区域的轮廓有重合
        if cv2.countNonZero(intersection) > 0:
            # 将 contour_original 转换为元组
            contour_original_tuple = tuple(map(tuple, contour_original.reshape(-1, 2)))
            
            # 将去除元件区域的轮廓和相关的原始轮廓存储在字典中
            if contour_without_tuple not in contour_relations:
                contour_relations[contour_without_tuple] = []
            contour_relations[contour_without_tuple].append(contour_original_tuple)
# 输出轮廓的从属关系
for contour_without, related_contours in contour_relations.items():
    print(f"Contour without components: {contour_without}")
    for related_contour in related_contours:
        print(f"  Related contour in original image: {related_contour}")
valid_contours = []  # 存储有效的连线轮廓
line_image = np.copy(image)  # 复制图像用于绘制

for contour in contours_without_components:
    if True :  # 过滤掉面积过小的轮廓
        # 创建掩膜来检测连线与元件区域的关系
        mask_with_contour = np.zeros_like(full_mask_padding)
        cv2.drawContours(mask_with_contour, [contour], -1, 255, thickness=cv2.FILLED)
        # 计算连线与元件区域的交集
        intersection_mask = cv2.bitwise_and(mask_with_contour, full_mask_padding)
        
        if cv2.countNonZero(intersection_mask) > 0:  # 如果有交集，说明是有效的连线
            valid_contours.append(contour)  # 添加有效连线
            cv2.drawContours(line_image, [contour], -1, (0, 255, 0), 2)  # 绘制有效连线
# print(valid_contours)
dilated_edges = cv2.dilate(edges_without_components, None, iterations=1)

# # 查找膨胀后的轮廓
# dilated_contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # 绘制有效连线
# line_image = np.copy(image)  # 复制图像用于绘制
# for contour in dilated_contours:
#     cv2.drawContours(line_image, [contour], -1, (0, 255, 0), 2)  # 绘制有效连线


# 原图
plt.subplot(1, 5, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 边缘检测结果
plt.subplot(1, 5, 2)
plt.title("Edges Detected")
plt.imshow(edges, cmap='gray')
plt.axis('off')

# 膨胀后的边缘图像
plt.subplot(1, 5, 3)
plt.title("Dilated Edges")
plt.imshow(dilated_edges, cmap='gray')
plt.axis('off')

# 去除元件区域后的边缘图像
plt.subplot(1, 5, 4)
plt.title("Edges Without Components")
plt.imshow(edges_without_components, cmap='gray')
plt.axis('off')

# 连线结果（绘制的连线）
plt.subplot(1, 5, 5)
plt.title("Detected Lines (Valid)")
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存结果
cv2.imwrite("edges_detected.png", edges)
cv2.imwrite("dilated_edges.png", dilated_edges)
cv2.imwrite("edges_without_components.png", edges_without_components)
cv2.imwrite("valid_line_image.png", line_image)