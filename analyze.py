import json,os,datetime,shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

script_directory = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR=os.path.join(script_directory,"6th_integrated_circuit_eda_elite_challenge_question10_dataset/all_images")

DATANO=38

DEFAULT_DILATED_ITERATION=2

DILATED_MAX_ITERATION=3

PADDING_NUM=1

LABEL_DICT = {
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
    'npn': 'NPN',
    'npn-cross': 'NPN',
    'bjt-npn-cross': 'NPN',
    'bjt-pnp': 'PNP',
    'pnp': 'PNP',
    'bjt-pnp-cross': 'PNP',
    'pnp-cross': 'PNP',
    'diode': 'Diode',
    'diso-amp': 'Diso_amp',
    'single-end-amp': 'Diso_amp',
    'siso-amp': 'Siso_amp',
    'single-input-single-end-amp': 'Siso_amp',
    'dido-amp': 'Dido_amp',
    'diff-amp': 'Dido_amp',
    'capacitor': 'Cap',
    'gnd': 'GND',
    'vdd': 'VDD',
    'inductor': 'Ind',
    'resistor': 'Res',
    'resistor-1': 'Res',
    'resistor-2': 'Res',
    'resistor2': 'Res',

    'cross-line-curved':'cross-line-curved',
    'port': 'port',
    'switch':'switch',
    'switch3':'switch',
    'switch-3':'switch',
}

RELATIVE_PORT_DICT={
    'pmos': {'Drain':(-1,-1), 'Source':(-1,1), 'Gate':(1,0)},
    'pmos-normal': {'Drain':(-1,-1), 'Source':(-1,1), 'Gate':(1,0)},
    'pmos-cross': {'Drain':(-1,-1), 'Source':(-1,1), 'Gate':(1,0),'Gate2':(-1,0)},
    'pmos-bulk': {'Drain':(-1,-1), 'Source':(-1,1), 'Gate':(1,0), 'Body':(-1,0)},
    'nmos': {'Drain':(-1,-1), 'Source':(-1,1), 'Gate':(1,0)},
    'nmos-normal': {'Drain':(-1,-1), 'Source':(-1,1), 'Gate':(1,0)},
    'nmos-cross': {'Drain':(-1,-1), 'Source':(-1,1), 'Gate':(1,0),'Gate2':(-1,0)},
    'nmos-bulk': {'Drain':(-1,-1), 'Source':(-1,1), 'Gate':(1,0), 'Body':(-1,0)},
    'voltage': {'Positive':(1,0), 'Negative':(-1,0)},# not accurate
    'voltage-1': {'Positive':(1,0), 'Negative':(-1,0)},
    'voltage-2': {'Positive':(1,0), 'Negative':(-1,0)},
    'voltage-lines': {'Positive':(1,0), 'Negative':(-1,0)},
    'current': {'In':(-1,0), 'Out':(1,0)},
    'bjt-npn': {'Base':(1,0), 'Emitter':(-1,1), 'Collector':(-1,-1)},
    'npn': {'Base':(1,0), 'Emitter':(-1,1), 'Collector':(-1,-1)},
    'bjt-npn-cross': {'Base':(1,0), 'Emitter':(-1,1), 'Collector':(-1,-1),'Base2':(-1,0)},
    'npn-cross': {'Base':(1,0), 'Emitter':(-1,1), 'Collector':(-1,-1),'Base2':(-1,0)},
    'bjt-pnp':  {'Base':(1,0), 'Emitter':(-1,1), 'Collector':(-1,-1)},
    'pnp':  {'Base':(1,0), 'Emitter':(-1,1), 'Collector':(-1,-1)},
    'bjt-pnp-cross': {'Base':(1,0), 'Emitter':(-1,1), 'Collector':(-1,-1),'Base2':(-1,0)},
    'pnp-cross': {'Base':(1,0), 'Emitter':(-1,1), 'Collector':(-1,-1),'Base2':(-1,0)},
    'diode': {'In':(1,0), 'Out':(-1,0)},
    'diso-amp': {'InN':(1,-0.5), 'InP':(1,0.5), 'Out':(-1,0)},# not accurate
    'single-end-amp': {'InN':(1,-0.5), 'InP':(1,0.5), 'Out':(-1,0)},# not accurate
    'siso-amp': {'In': (1,0), 'Out':(-1,0)},
    'single-input-single-end-amp': {'In': (1,0), 'Out':(-1,0)},
    'dido-amp': {'InN':(1,-0.5), 'InP':(1,0.5), 'OutN':(-1,-0.5), 'OutP':(-1,0.5)},# not accurate
    'diff-amp': {'InN':(1,-0.5), 'InP':(1,0.5), 'OutN':(-1,-0.5), 'OutP':(-1,0.5)},# not accurate
    'capacitor': {'Pos': (1,0), 'Neg':(-1,0)},
    'gnd': {'port': (1,0)},
    'vdd': {'port': (-1,0)},
    'inductor': {'Pos': (1,0), 'Neg':(-1,0)},
    'resistor': {'Pos': (1,0), 'Neg':(-1,0)},
    'resistor-1': {'Pos': (1,0), 'Neg':(-1,0)},
    'resistor-2': {'Pos': (1,0), 'Neg':(-1,0)},
    'resistor2': {'Pos': (1,0), 'Neg':(-1,0)},

    'cross-line-curved':{'Up':(0,1),'Down':(0,-1),'Left':(-1,0),'Right':(1,0)},
    'port': {'port':(1,0)},
    'switch':{'port1':(1,0),'port2':(0,1)},
    'switch3':{'port1':(1,0),'port2':(0,1),'port3':(0,1)},
    'switch-3':{'port1':(1,0),'port2':(0,1),'port3':(0,1)},
}

PORT_DICT={}
for key in RELATIVE_PORT_DICT:
    PORT_DICT[key]={key1:None for key1 in RELATIVE_PORT_DICT[key]}


class UnionFind:
    def __init__(self, n):
        # 初始化父亲数组，每个结点的父结点指向自己
        self.parent = list(range(n))  
        # 用来记录每个集合的大小（用于优化路径压缩）
        self.rank = [1] * n  
    
    def find(self, x):
        # 路径压缩，查找 x 的根结点
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        # 合并操作，使用按秩合并优化
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            # 让较小的树挂到较大的树下面，保持树的平衡
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


class annotated_units:
    def __init__(self,image,annotation_data):
        self.image=image
        self.annotation_data=annotation_data
        # 获取图像尺寸
        self.height, self.width = image.shape[:2]
        self.unit_info=[]
        for idx,shape in enumerate(self.annotation_data["shapes"]):
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]
            label = shape['label']
            self.unit_info.append({'idx':idx,
                                   'field':[x1,y1,x2,y2],
                                   'label':label,
                                   'related_contours':[]
                                   })


        # 转换为灰度图
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用高斯模糊减少噪声
        self.blurred_image = cv2.GaussianBlur(self.gray_image, (5, 5), sigmaX=1)
        # 使用Canny边缘检测
        self.edges = cv2.Canny(self.blurred_image, 0, 150)
        # 创建空白掩膜full_mask，初始化为全0
        self.full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for idx,shape in enumerate(self.annotation_data["shapes"]):
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 创建一个PIL图像对象来绘制矩形框
            mask = Image.new('L', (self.width, self.height), 0)  # 创建黑色背景（全0掩膜）
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)  # 填充矩形区域为白色
            # 将PIL图像转回为NumPy数组
            component_mask = np.array(mask)
            self.unit_info[idx]['component_mask']=component_mask
            self.full_mask = cv2.bitwise_or(self.full_mask, component_mask)

        #full_mask_padding
        padding=PADDING_NUM
        self.full_mask_padding = np.zeros((self.height, self.width), dtype=np.uint8)
        for idx,shape in enumerate(self.annotation_data["shapes"]):
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]

            mask_padding = Image.new('L', (self.width, self.height), 0) 
            draw = ImageDraw.Draw(mask_padding)
            draw.rectangle([x1-padding, y1-padding, x2+padding, y2+padding], fill=255)  
            component_mask_padding = np.array(mask_padding)
            self.unit_info[idx]['component_mask_padding']=component_mask_padding
            self.full_mask_padding = cv2.bitwise_or(self.full_mask_padding, component_mask_padding)

        # 反转掩膜，得到框外区域
        self.inverse_mask = cv2.bitwise_not(self.full_mask)

        # 使用掩膜和边缘图像，去除元件区域
        self.edges_without_components = cv2.bitwise_and(self.edges, self.edges, mask=self.inverse_mask)

        # 查找轮廓
        self.contours_original, _ = cv2.findContours(self.edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours_without_components, _ = cv2.findContours(self.edges_without_components, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def get_contour_relation(self,slave_contours, master_contours):# 没用到
        contour_relations = {}
        for slave_contour in slave_contours:
            # 创建一个掩膜来绘制当前的轮廓
            mask_without = np.zeros_like(self.edges, dtype=np.uint8)
            cv2.drawContours(mask_without, [slave_contour], -1, 255, thickness=cv2.FILLED)
            
            # 将 slave_contour 转换为元组（不可变类型），作为字典的键
            slave_contour_tuple = tuple(map(tuple, slave_contour.reshape(-1, 2)))
            
            # 遍历原始图像中的轮廓，判断哪些轮廓在当前轮廓的区域内
            for master_contour in master_contours:
                mask_original = np.zeros_like(self.edges, dtype=np.uint8)
                cv2.drawContours(mask_original, [master_contour], -1, 255, thickness=cv2.FILLED)

                # 计算交集
                intersection = cv2.bitwise_and(mask_without, mask_original)
                
                # 如果交集区域非零，说明原始轮廓与去除元件区域的轮廓有重合
                if cv2.countNonZero(intersection) > 0:
                    # 将 master_contour 转换为元组
                    master_contour_tuple = tuple(map(tuple, master_contour.reshape(-1, 2)))
                    
                    # 将去除元件区域的轮廓和相关的原始轮廓存储在字典中
                    if slave_contour_tuple not in contour_relations:
                        contour_relations[slave_contour_tuple] = []
                    contour_relations[slave_contour_tuple].append(master_contour_tuple)
        return contour_relations
        # # 输出轮廓的从属关系
        # for slave_contour, related_contours in contour_relations.items():
        #     print(f"Contour without components: {slave_contour}")
        #     for related_contour in related_contours:
        #         print(f"  Related contour in original image: {related_contour}")
    
    def get_valid_contours_info(self,contours):
        valid_contours = []  # 存储有效的连线轮廓
        valid_related_unit_info=[]
        for contour in contours:
            # 创建掩膜来检测连线与元件区域的关系
            mask_with_contour = np.zeros_like(self.full_mask_padding)
            cv2.drawContours(mask_with_contour, [contour], -1, 255, thickness=cv2.FILLED)
            contour_port_cnt=0
            related_unit_info=[]
            for unit in self.unit_info:
                # 计算连线与元件区域的交集
                intersection_mask = cv2.bitwise_and(mask_with_contour, unit['component_mask_padding'])# row, column
                non_zero_points = cv2.findNonZero(intersection_mask) # x,y  and  column,row
                if non_zero_points is not None:
                    # 计算非零点的平均值（即中心点）
                    center1 = np.mean(non_zero_points, axis=0)  # axis=0 对每列求平均
                    center = tuple(np.round(center1[0]).astype(int))  # 转换为元组 (x, y)
                    related_unit_info.append({'unit_idx':unit['idx'],'port_pos':center})#TODO can consider one contour maps to multi-port
                    # if unit['idx']==7:
                    #     print(center1,center,non_zero_points)
                    contour_port_cnt+=1
            if contour_port_cnt >=2:
                valid_contours.append(contour)
                valid_related_unit_info.append(related_unit_info)
        return (valid_contours,valid_related_unit_info)
    
    def get_valid_contours_port_num(self,contours):# take less resources
        total_num=0
        for contour in contours:
            # 创建掩膜来检测连线与元件区域的关系
            mask_with_contour = np.zeros_like(self.full_mask_padding)
            cv2.drawContours(mask_with_contour, [contour], -1, 255, thickness=cv2.FILLED)
            contour_port_cnt=0
            for unit in self.unit_info:
                # 计算连线与元件区域的交集
                intersection_mask = cv2.bitwise_and(mask_with_contour, unit['component_mask_padding'])
                if cv2.countNonZero(intersection_mask) > 0:  # 如果有交集，说明是有效的连线
                    contour_port_cnt+=1
            if contour_port_cnt >=2:
                total_num+=contour_port_cnt
        return total_num
    
    def show_contours_image(self, contours, related_unit_info=None, annotate_unit_label: bool = False):
        '''
            Similar to show_netlist_image, but for contours and annotations.
        '''
        image = self.image.copy()  # 复制图像，避免修改原图
        fig, ax = plt.subplots()

        # 显示原图，仅调用一次
        ax.imshow(image)  # 显示原图

        # 绘制轮廓
        for contour in contours:
            # 使用 matplotlib 绘制轮廓，转化为 polygon
            contour = np.array(contour).reshape((-1, 2))
            ax.plot(contour[:, 0], contour[:, 1], color='green', linewidth=2)

        if related_unit_info is not None:
            for unit_info_piece in related_unit_info:
                for port_info in unit_info_piece:
                    port_pos = port_info['port_pos']
                    # 绘制端口位置（红色圆点标记端口）
                    ax.plot(port_pos[0], port_pos[1], 'ro', markersize=5)  # 'ro' 表示红色圆点

        if annotate_unit_label:
            for unit in self.unit_info:
                # 提取矩形框坐标
                x1, y1, x2, y2 = unit['field']
                # 确保坐标是整数
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = unit['label']
                idx = unit['idx']
                color = (255, 0, 0)  # 红色文本

                # 绘制矩形框
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none'))

                # 在矩形框上方添加文本标签
                ax.text(x1, y1 - 10, f'{idx}.{label}', color='red', fontsize=8)

        # 设置标题和轴
        ax.set_title('Contours and Units')
        ax.axis('off')  # 隐藏坐标轴

        # 显示图像
        plt.show()
    def get_contours(self,if_search_iter:bool=False):#考虑图像分辨率不同,典型线宽不同的影响,找到迭代数
        if if_search_iter:#TODO ....
            minimum_port_num=self.get_valid_contours_port_num(self.contours_without_components)
            self.valid_contours_port_num=minimum_port_num
            self.dilated_iteration=0
            for i in range(1,DILATED_MAX_ITERATION+1):#TODO to ...num of ports should have
                dilated_edges=cv2.dilate(self.edges, None, iterations=i)
                dilated_edges_without_component=cv2.bitwise_and(dilated_edges, dilated_edges, mask=self.inverse_mask)
                contours, _ = cv2.findContours(dilated_edges_without_component, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                port_num=self.get_valid_contours_port_num(contours)

                _,port_info=self.get_valid_contours_info(contours)

                print(port_num)
                self.show_contours_image(contours,port_info)
                if(port_num<minimum_port_num):
                    self.dilated_iteration=i
                    self.valid_contours_port_num=port_num
        else:
            self.dilated_iteration=DEFAULT_DILATED_ITERATION

        self.dilated_edges=cv2.dilate(self.edges, None, iterations=self.dilated_iteration)
        self.dilated_edges_without_component=cv2.bitwise_and(self.dilated_edges, self.dilated_edges, mask=self.inverse_mask)

        dilated_contours_without_components, _ = cv2.findContours(self.dilated_edges_without_component, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.valid_contours_info=self.get_valid_contours_info(dilated_contours_without_components)
        for i in range(len(self.valid_contours_info[1])):
            for port_info in self.valid_contours_info[1][i]:
                self.unit_info[port_info['unit_idx']]['related_contours'].append({'contour_idx':i,'port_pos':port_info['port_pos']})

    def del_unit_list_component_mask(self,unit_info=None):
        if unit_info is None:
            for unit in self.unit_info:
                unit.pop('component_mask_padding',None)
                unit.pop('component_mask',None)
            return self.unit_info
        else:
            unit_info=unit_info.copy()
            for unit in unit_info:
                unit.pop('component_mask_padding',None)
                unit.pop('component_mask',None)
            return unit_info

    def print_unit_info(self,unit_info=None):
        if unit_info is None:
            a=self.del_unit_list_component_mask(self.unit_info)
            print(a)
        else:
            a=self.del_unit_list_component_mask(unit_info)
            print(a)

                
    def calculate_local_symmetry(self,image, window_size=5):
        """
        计算图像的局部对称性，通过滑动窗口计算每个像素点附近的平均值差异。
        window_size：滑动窗口的大小，默认5x5。
        返回对称性得分：值越小表示越对称。
        """
        # 获取图像的尺寸
        height, width = image.shape

        # 执行图像上下翻转和左右翻转
        flip_up_down = np.flipud(image)
        flip_left_right = np.fliplr(image)

        # 使用滑动窗口计算局部区域的对称性
        half_window = window_size // 2
        symmetry_up_down = np.zeros_like(image, dtype=np.float32)
        symmetry_left_right = np.zeros_like(image, dtype=np.float32)

        # 对每个像素点周围的窗口进行处理
        for i in range(half_window, height - half_window):
            for j in range(half_window, width - half_window):
                # 提取局部窗口
                window_original = image[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                window_flip_up_down = flip_up_down[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                window_flip_left_right = flip_left_right[i-half_window:i+half_window+1, j-half_window:j+half_window+1]

                # 计算窗口内的均值
                mean_original = np.mean(window_original)
                mean_flip_up_down = np.mean(window_flip_up_down)
                mean_flip_left_right = np.mean(window_flip_left_right)

                # 计算差异，存储到对称性得分中
                symmetry_up_down[i, j] = np.abs(mean_original - mean_flip_up_down)
                symmetry_left_right[i, j] = np.abs(mean_original - mean_flip_left_right)

        # 对所有像素的对称性得分求和，得到总体的对称性得分
        symmetry_up_down_score = np.sum(symmetry_up_down)
        symmetry_left_right_score = np.sum(symmetry_left_right)

        return symmetry_up_down_score, symmetry_left_right_score

    def find_symmetry_and_density(self, unit, window_size=5, tell_density:bool=True):
            """
            根据元件范围分析上下左右对称性，并计算对称轴两侧的白度密度
            """
            # 获取元件的范围（field），并提取出相应区域的图像
            unit_range = unit['field']
            x1, y1, x2, y2 = map(int, unit_range)  # 将范围转换为整数

            # 从原始图像中裁剪出元件区域
            unit_image = self.image[y1:y2, x1:x2]

            # 确保图像是灰度图
            gray_image = cv2.cvtColor(unit_image, cv2.COLOR_BGR2GRAY) if len(unit_image.shape) == 3 else unit_image

            aligned_image = gray_image  # TODO can make image more focus on the main body to avoid the impact of text

            result={}

            # 计算局部对称性得分
            symmetry_up_down_score, symmetry_left_right_score = self.calculate_local_symmetry(aligned_image, window_size)

            # 判断上下对称性和左右对称性哪个更好
            if symmetry_up_down_score < symmetry_left_right_score:
                symmetry_direction = "Vertical"
                symmetry_value = symmetry_up_down_score
                axis_position = (y1+y2) // 2  # 垂直轴的位置是图像的中心行
                another_mid_axis_position = (x1+x2) // 2
                is_vertical = True
            else:
                symmetry_direction = "Horizontal"
                symmetry_value = symmetry_left_right_score
                axis_position = (x1+x2) // 2  # 水平轴的位置是图像的中心列
                another_mid_axis_position = (y1+y2) // 2
                is_vertical = False
            result['is_vertical'] = is_vertical
            result["symmetry_direction"] = symmetry_direction
            result["symmetry_value"] = symmetry_value

            # 查找最接近对称轴的端口
            closest_port = None
            min_distance = float('inf')
            for contour_info in unit['related_contours']:
                port_pos = contour_info['port_pos']
                port_y = port_pos[1]  # 端口的纵坐标
                port_x = port_pos[0]  # 端口的横坐标

                # 计算端口位置与对称轴的距离
                if is_vertical:
                    # 如果选择了垂直对称轴，则用端口的纵坐标来确定
                    distance = abs(port_y - axis_position)
                else:
                    # 如果选择了水平对称轴，则用端口的横坐标来确定
                    distance = abs(port_x - axis_position)

                if distance < min_distance:
                    min_distance = distance
                    closest_port = port_pos

            result["closest_port"] = closest_port

            # 根据端口位置构建新的对称轴
            if is_vertical:
                axis_position = closest_port[1] if closest_port is not None else axis_position# 竖直对称轴根据端口的纵坐标来确定
            else:
                axis_position = closest_port[0] if closest_port is not None else axis_position# 水平对称轴根据端口的横坐标来确定
            result["axis_position"] = axis_position
            result['another_mid_axis_position'] = another_mid_axis_position

            # 根据选择的对称性方向计算对称轴两侧的白度密度
            if tell_density:
                if is_vertical:
                    # 选择垂直方向（上下对称性）对称轴
                    top_side = aligned_image[:axis_position-y1, :]  # 图像的上半部分
                    bottom_side = aligned_image[axis_position-y1+1:, :]  # 图像的下半部分
                    left_side = aligned_image[:, :another_mid_axis_position-x1]  # 图像的左半部分
                    right_side = aligned_image[:, another_mid_axis_position-x1+1:]  # 图像的右半部分
                else:
                    # 选择水平方向（左右对称性）对称轴
                    top_side = aligned_image[:another_mid_axis_position-y1, :]  # 图像的上半部分
                    bottom_side = aligned_image[another_mid_axis_position-y1+1:, :]  # 图像的下半部分
                    left_side = aligned_image[:, :axis_position-x1]  # 图像的左半部分
                    right_side = aligned_image[:, axis_position-x1+1:]  # 图像的右半部分

                top_density = np.mean(top_side)  # 上半部分的白度密度
                bottom_density = np.mean(bottom_side)  # 下半部分的白度密度
                left_density = np.mean(left_side)  # 左半部分的白度密度
                right_density = np.mean(right_side)  # 右半部分的白度密度

                # 计算白度密度
                if is_vertical:
                    d1=top_density
                    d2=bottom_density
                    d3=left_density
                    d4=right_density
                    higher_density_side = "Top" if top_density > bottom_density else "Bottom"
                    ano_higher_density_side = "Left" if left_density > right_density else "Right"
                    if top_density < bottom_density :
                        p2=np.array([0,1])
                    else:
                        p2=np.array([0,-1])
                    if left_density > right_density :
                        p1=np.array([1,0])
                    else:
                        p1=np.array([-1,0])

                else:
                    d1=left_density
                    d2=right_density
                    d3=top_density
                    d4=bottom_density
                    higher_density_side = "Left" if left_density > right_density else "Right"
                    ano_higher_density_side = "Top" if top_density > bottom_density else "Bottom"
                    if top_density < bottom_density :
                        p1=np.array([0,1])
                    else:
                        p1=np.array([0,-1])
                    if left_density > right_density :
                        p2=np.array([1,0])
                    else:
                        p2=np.array([-1,0])

                result['d1'] = d1
                result['d2'] = d2
                result['d3'] = d3
                result['d4'] = d4
                result['p1']=p1
                result['p2']=p2                
                result["higher_density_side"] = higher_density_side
                result['ano_higher_density_side'] = ano_higher_density_side
            return result

    def plot_symmetry_with_ports(self,image, unit, axis_position, is_vertical, symmetry_direction):
        """
        绘制图像，标注元件的范围、label、端口位置，并标出对称轴及最接近的端口。
        
        :param image: 输入的图像
        :param unit: 包含元件信息的字典，结构如 {'idx': 7, 'field': [x_min, y_min, x_max, y_max], 'label': 'pmos-cross', 'related_contours': [{'contour_idx': 8, 'port_pos': (x, y)}, ...]}
        :param axis_position: 对称轴的位置（纵坐标或横坐标）
        :param is_vertical: 是否是垂直对称轴（True：上下对称，False：左右对称）
        :param symmetry_direction: 对称方向（"Vertical" 或 "Horizontal"）
        """
        # 转换为灰度图像以便显示
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        # 创建图像和坐标轴
        plt.figure(figsize=(8, 8))
        plt.imshow(gray_image)
        
        # 画出元件范围（矩形框）
        x_min, y_min, x_max, y_max = unit['field']
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        
        # 标注label
        plt.text(x_min, y_min - 10, unit['label'], color='red', fontsize=12, ha='center')
        
        # 标出所有端口位置
        port_positions = [contour['port_pos'] for contour in unit['related_contours']]
        ports = np.array(port_positions)
        plt.scatter(ports[:, 0], ports[:, 1], color='blue', label='Ports', zorder=5)
        
        # 高亮最接近对称轴的端口
        closest_port = None
        min_distance = float('inf')
        for port_pos in port_positions:
            distance = abs(port_pos[1] - axis_position) if is_vertical else abs(port_pos[0] - axis_position)
            if distance < min_distance:
                min_distance = distance
                closest_port = port_pos
                
        if closest_port is not None:
            plt.scatter(closest_port[0], closest_port[1], color='green', s=100, edgecolor='black', label='Closest Port', zorder=5)
        
        # 绘制对称轴
        if is_vertical:
            plt.axhline(y=axis_position, color='yellow', linestyle='--', label='Symmetry Axis')
        else:
            plt.axvline(x=axis_position, color='yellow', linestyle='--', label='Symmetry Axis')
        
        # 标注对称方向
        plt.text(image.shape[1] // 2, image.shape[0] - 20, f'Symmetry Direction: {symmetry_direction}', color='yellow', fontsize=14, ha='center')
        
        # 显示图像和标注
        plt.legend(loc='upper left')
        plt.axis('off')
        plt.show()

    def plot_test(self):
        self.del_unit_list_component_mask()
        # for i in range(7,15):
        #     dict1=self.find_symmetry_and_density(self.unit_info[i])
        #     print(dict1)
        #     self.plot_symmetry_with_ports(self.image,self.unit_info[i],dict1['Axis Position'],dict1['is_vertical'],dict1['Symmetry Direction'])

    def find_nearest_port(self, relative_x, relative_y, unit):
        """
        找到与给定相对位置 (relative_x, relative_y) 最近的端口对应的 related_contour。

        Parameters:
        - relative_x: 在 unit['field'] 中的相对 x 位置 (-1~1)
        - relative_y: 在 unit['field'] 中的相对 y 位置 (-1~1)
        - unit: 包含 unit 数据结构的字典

        Returns:
        - 最近的端口对应的 related_contour :{'contour_idx':i,'port_pos':port_info['port_pos']}
        """
        # 获取 unit 的 field 坐标 (left, top, right, bottom)
        field_left, field_top, field_right, field_bottom = unit['field']
        
        # 将相对坐标转换为绝对坐标
        absolute_x = ((field_left+field_right) + relative_x * (field_right - field_left))/2
        absolute_y = ((field_top +field_bottom) - relative_y * (field_bottom - field_top))/2
        
        # 初始化最小距离和最近的 related_contour
        min_distance = float('inf')
        nearest_contour = None

        # 遍历 unit['related_contours'] 中的所有端口
        for contour in unit['related_contours']:
            port_pos = contour['port_pos']  # 端口位置
            port_x, port_y = port_pos[0], port_pos[1]
            
            # 计算与端口的欧氏距离
            distance = (absolute_x - port_x) ** 2 + (absolute_y - port_y) ** 2
            
            # 更新最小距离和最近的 related_contour
            if distance < min_distance:
                min_distance = distance
                nearest_contour = contour

        # 返回最近的 related_contour
        return nearest_contour

    def tramsform_relative_port(self,p1,p2,relative_ports:dict):
        cos_theta=p1[0]
        sin_theta=p1[1]
        rotate=np.array([[cos_theta,-sin_theta],[sin_theta,cos_theta]])
        reverse=np.array([[1,0],[0,-1]])
        if np.array_equal(p2,  np.array([-sin_theta,cos_theta])):
            if_reverse=False
        elif np.array_equal(p2 , -np.array([-sin_theta,cos_theta])):
            if_reverse=True
        else:
            raise ValueError
        if if_reverse:
            transform=np.dot(rotate,reverse)
        else:
            transform=rotate
        
        results={}
        for relative_port in relative_ports:
            results[relative_port]=np.dot(transform,np.array(relative_ports[relative_port]))

        return results

    def get_single_unit_contour_connnections(self,unit):
        componentname=unit['label']
        final_lable=LABEL_DICT.get(componentname , None)
        contour_connection=PORT_DICT.get(componentname , None)
        if contour_connection is None:
            print(f'PORT_DICT:unknown labelname of component: {componentname}')
        else: 
            contour_connection=contour_connection.copy()
        
        if  final_lable is None:
            print(f'LABEL_DICT:unknown labelname of component: {componentname}')
        else:
            ports_relative_pos=RELATIVE_PORT_DICT[componentname]
            if componentname!='cross-line-curved':
                symmetry=self.find_symmetry_and_density(unit)
                p1=symmetry['p1']
                p2=symmetry['p2']
                ports_relative_pos=self.tramsform_relative_port(p1,p2,RELATIVE_PORT_DICT[componentname])
                # print(ports_relative_pos)
            for key in ports_relative_pos:
                related_contour=self.find_nearest_port(ports_relative_pos[key][0],ports_relative_pos[key][1],unit)
                contour_connection[key]=related_contour
            # if componentname == 'current':
            #     print(unit,'\n',ports_relative_pos,contour_connection)
        unit['contour_connection']=contour_connection#unit['contour_connection][port_name]=related_contour:{'contour_idx':i,'port_pos':port_info['port_pos']}

    def get_all_unit_contour_connnections(self):
        for unit in self.unit_info:
            self.get_single_unit_contour_connnections(unit)

    def get_netlist(self,if_debug:bool=False):
        n=len(self.valid_contours_info[0])+2
        uf = UnionFind(n)
        equivalences=[]
        gnd_idx=n-2
        vdd_idx=n-1
        for unit in self.unit_info:
            componentname=unit['label']
            try:
                if  componentname in ['pmos-cross','nmos-cross']:
                    idx1=unit['contour_connection']['Gate']['contour_idx']
                    idx2=unit['contour_connection']['Gate2']['contour_idx']
                    equivalences.append((idx1,idx2))
                elif componentname in ['bjt-npn-cross','bjt-pnp-cross']:
                    idx1=unit['contour_connection']['Base']['contour_idx']
                    idx2=unit['contour_connection']['Base2']['contour_idx']
                    equivalences.append((idx1,idx2))
                elif componentname == 'cross-line-curved':
                    idx1=unit['contour_connection']['Up']['contour_idx']
                    idx2=unit['contour_connection']['Down']['contour_idx']
                    idx3=unit['contour_connection']['Left']['contour_idx']
                    idx4=unit['contour_connection']['Right']['contour_idx']
                    equivalences.append((idx1,idx2))
                    equivalences.append((idx3,idx4))
                elif componentname == 'gnd':
                    idx=unit['contour_connection']['port']['contour_idx']
                    equivalences.append((idx,gnd_idx))
                elif componentname == 'vdd' :
                    idx=unit['contour_connection']['port']['contour_idx']
                    equivalences.append((idx,vdd_idx))
            except:
                continue

        for x, y in equivalences:
            uf.union(x, y)
        all_net = {}
        for i in range(n):
            root = uf.find(i)
            if root not in all_net:
                all_net[root] = []
            all_net[root].append(i)

        net_num=1

        self.contour2netlist={}
        for net in all_net.values():
            if gnd_idx in net:
                for contour_idx in net:
                    self.contour2netlist[contour_idx]='GND'
            elif vdd_idx in net:
                for contour_idx in net:
                    self.contour2netlist[contour_idx]='VDD'
            else:
                for contour_idx in net:
                    self.contour2netlist[contour_idx]=f'net{net_num}'
                net_num+=1
        
        for unit in self.unit_info:
            componentname=unit['label']
            final_lable=LABEL_DICT.get(componentname , None)
            if final_lable is not None:
                try:
                    if if_debug:
                        unit['net']={'component_type':final_lable,'port_connection':{port_name:self.contour2netlist[related_contour['contour_idx']] for port_name,related_contour in unit['contour_connection'].items()},'port_pos':{port_name:related_contour['port_pos'] for port_name,related_contour in unit['contour_connection'].items()}}
                    else:
                        unit['net']={'component_type':final_lable,'port_connection':{port_name:self.contour2netlist[related_contour['contour_idx']] for port_name,related_contour in unit['contour_connection'].items()}}
                except:
                    if if_debug:
                        print(f"Warning: couldn't build net of unit")
                        print(f'warning unit :{unit}')

    def show_netlist_image(self):
        '''
            can be applied only after self.get_netlist(True) have been excuted
        '''
        image = self.image.copy()  # 复制图像，避免修改原图
        # 使用matplotlib绘图
        fig, ax = plt.subplots()
        # 假设原图已加载为numpy数组，显示图像
        ax.imshow(self.image)  # 显示原图

        for unit in self.unit_info:
            net_info = unit['net']  # {'component_type': 'Voltage', 'port_connection': {'Positive': 'net2', 'Negative': 'GND'}, 'port_pos': {'Positive': (92, 164), 'Negative': (55, 163)}}
            port_pos = net_info['port_pos']  # {'Positive': (92, 164), 'Negative': (55, 163)}
            port_connection = net_info['port_connection']  # {'Positive': 'net2', 'Negative': 'GND'}

            # 标记端口位置和添加文字
            for port, pos in port_pos.items():
                # 绘制端口位置（红色圆点标记端口）
                ax.plot(pos[0], pos[1], 'ro', markersize=5)  # 'ro' 表示红色圆点
                
                # 添加文字说明，显示端口和连接信息
                connection_name = port_connection[port]  # 获取端口的连接信息（如 'net2' 或 'GND'）
                ax.text(pos[0] , pos[1] - 10, f"{port}: {connection_name}", color='blue', fontsize=8)

            # 设置标题和轴
        ax.set_title('Ports and Connections')
        ax.axis('off')  # 隐藏坐标轴

        # 显示图像
        plt.show()

    def show_netlist_and_contours(self, contours=None, related_unit_info=None, annotate_unit_label: bool = True,extrernal_title=None):
        '''
            Displays netlist and contours in two side-by-side panels.
        '''
        if contours is None:
            contours=self.valid_contours_info[0]
        if related_unit_info is None:
            related_unit_info=self.valid_contours_info[1]
        image = self.image.copy()  # 复制图像，避免修改原图

        # 创建两个子图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 两个子图并排显示

        # 第一个子图：显示 netlist 信息
        ax1 = axes[0]
        ax1.imshow(image)  # 显示原图
        ax1.set_title('Netlist Ports and Connections')
        ax1.axis('off')  # 隐藏坐标轴
        # 在左上角添加文件名
        if extrernal_title is not None:
            ax1.text(
                10, 10, f"File: {extrernal_title}",
                color='black', fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8)
            )
        for unit in self.unit_info:
            try:
                net_info = unit['net']  # 获取每个unit的net信息
                port_pos = net_info['port_pos']  # 端口位置字典
                port_connection = net_info['port_connection']  # 端口连接信息字典

                # 标记端口位置和添加文字
                for port, pos in port_pos.items():
                    # 绘制端口位置（红色圆点标记端口）
                    ax1.plot(pos[0], pos[1], 'ro', markersize=5)  # 'ro' 表示红色圆点

                    # 添加文字说明，显示端口和连接信息
                    connection_name = port_connection[port]  # 获取端口的连接信息
                    ax1.text(pos[0], pos[1] - 10, f"{port}: {connection_name}", color='blue', fontsize=8)
            except:
                continue

        # 第二个子图：显示 contours 和注释信息
        ax2 = axes[1]
        ax2.imshow(image)  # 显示原图
        ax2.set_title('Contours and Annotations')
        ax2.axis('off')  # 隐藏坐标轴

        # 绘制轮廓
        for contour in contours:
            # 使用 matplotlib 绘制轮廓
            contour = np.array(contour).reshape((-1, 2))
            ax2.plot(contour[:, 0], contour[:, 1], color='green', linewidth=2)

        if related_unit_info is not None:
            for unit_info_piece in related_unit_info:
                for port_info in unit_info_piece:
                    port_pos = port_info['port_pos']
                    # 绘制端口位置（红色圆点标记端口）
                    ax2.plot(port_pos[0], port_pos[1], 'ro', markersize=5)  # 'ro' 表示红色圆点

        if annotate_unit_label:
            for unit in self.unit_info:
                # 提取矩形框坐标
                x1, y1, x2, y2 = unit['field']
                # 确保坐标是整数
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = unit['label']
                idx = unit['idx']

                # 绘制矩形框
                ax2.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none'))

                # 在矩形框上方添加文本标签
                ax2.text(x1, y1 - 10, f'{idx}.{label}', color='red', fontsize=8)

        # 调整子图布局
        plt.tight_layout()
        plt.show()
        
    def analyze(self,extrernal_title=None,if_debug=False):
        self.get_contours()
        self.del_unit_list_component_mask()
        self.get_all_unit_contour_connnections()
        self.get_netlist(if_debug)
        if if_debug:
            for unit in self.unit_info:
                print(unit)
            print(self.contour2netlist)
            self.show_netlist_and_contours(extrernal_title=extrernal_title)

def create_run_dir(base_dir, prefix="run", timestamp=True):
    """
    创建一个运行目录，支持自定义前缀和时间戳。
    
    Args:
        base_dir (str): 运行目录的父目录。
        prefix (str): 运行目录的前缀名称，默认 "run"。
        timestamp (bool): 是否在目录名中添加时间戳，默认 True。
    
    Returns:
        str: 创建的运行目录的完整路径。
    """
    # 确保基础目录存在
    os.makedirs(base_dir, exist_ok=True)
    
    # 构造目录名称
    dir_name = prefix
    if timestamp:
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name += f"_{time_str}"
    
    # 生成完整路径
    run_dir = os.path.join(base_dir, dir_name)
    
    # 确保目录唯一性
    counter = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(base_dir, f"{dir_name}_{counter}")
        counter += 1
    
    # 创建目录
    os.makedirs(run_dir)
    print(f"Run directory created: {run_dir}")
    return run_dir


def task1(datano:int):
    '''
        only used on 6th_integrated_circuit_eda_elite_challenge_question10_dataset
        DATASET_DIR should be its directory
    '''
    json_path=DATASET_DIR+f"\\{datano}.json"
    image_path=DATASET_DIR+f"\\{datano}.png"

    with open(json_path, "r") as f:
        annotation_data = json.load(f)

    image = cv2.imread(image_path)  # 读取彩色图
    if image is None:
        raise FileNotFoundError(f"Image {image_path} not found!")

    #endregion
    try:
        example=annotated_units(image,annotation_data)
        example.analyze(f"{datano}.png",True)
        example.show_contours_image(example.valid_contours_info[0],example.valid_contours_info[1],True)
    except KeyboardInterrupt:
        exit(0)
    except:
        try:
            example.get_contours()
            example.show_contours_image(example.valid_contours_info[0],example.valid_contours_info[1],True)
        except:
            print(f'Picture NO.{datano} contours analysis failed !')
        print(f'Picture NO.{datano} netlist analysis failed !')

def convert_yolo_result_to_json(result):
    """
    将YOLO推理结果转换为指定格式的JSON。

    Args:
        result (ultralytics.yolo.engine.results.Results): YOLO推理结果对象。
        image_path (str): 对应图像的路径。

    Returns:
        dict: 转换后的JSON数据。
    """
    shapes = []
    image_height,image_width=result.orig_shape
    class_names=result.names
    # 获取推理结果中的边界框信息
    boxes = result.boxes
    for i in range(boxes.shape[0]):
        # 每个检测框的信息，包括类别、置信度、坐标等
        class_id = int(boxes.cls[i])  # 类别ID
        x_min, y_min, x_max, y_max = boxes.xyxy[i].tolist()  # 边界框坐标
        
        # 创建一个shape对象
        shape = {
            "label": class_names.get(class_id, "unknown"),
            "points": [[x_min, y_min], [x_max, y_max]],
        }
        shapes.append(shape)
    
    # 创建最终的JSON结构
    json_data = {
        "shapes": shapes,
        "imagePath": os.path.basename(result.path),
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    
    return json_data

def yolomodel_run(model_dir,image_dir,run_dir,generate_dir):
    from ultralytics import YOLO
    # Load a model
    model = YOLO(model_dir)  # pretrained YOLO11n model
    # Run batched inference on a list of images
    results = model(image_dir,save=True
                    # save_txt=True,project=temp_dir
                    )  # return a list of Results objects
    # Process results list
    if True:
        # cnt=0
        for result in results:
            annotation_data=convert_yolo_result_to_json(result)
            image=result.orig_img
            t=annotated_units(image,annotation_data)
            try:
                t.analyze()
                unit_info=t.unit_info
            except:
                unit_info=None
            # t.analyze(if_debug=True)
            del t
            ckt_netlist=[unit.get('net',None) for unit in unit_info if unit.get('net',None) is not None] if unit_info is not None else None
            ckt_type='DISO-Amplifier'
            final_dict={'ckt_netlist':ckt_netlist,
                        'ckt_type':ckt_type,
                        }
            generate_txt_path=os.path.join(generate_dir,os.path.splitext(os.path.basename(result.path))[0]+'.txt')
            with open(generate_txt_path,'w') as f:
                f.write(str(final_dict))
            print(f'finish analysis of {result.path}')
            # cnt+=1
            # if cnt>1:
            #     break
            
if __name__ == '__main__':
    # data = [38]
    # with ProcessPoolExecutor(max_workers=10) as executor:
    #     results = list(executor.map(task1, data))
    # task1(38)
    model_dir=os.path.join(script_directory,r"datasets\eda\yolo_train_run3\weights\best.pt")
    run_dir=os.path.join(script_directory,r"public\run")
    image_dir=os.path.join(script_directory,r'public\images')
    generate_dir=os.path.join(script_directory,r'public\generate1')
    os.makedirs(generate_dir,exist_ok=True)
    yolomodel_run(model_dir,image_dir,run_dir,generate_dir)
