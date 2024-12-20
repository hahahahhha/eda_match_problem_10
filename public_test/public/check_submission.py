import os
import re

def get_submission_folders(path):
    submission_folders = []  # 存储符合条件的文件夹名称

    # 遍历给定路径下的所有项目
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)  # 获取文件夹完整路径
        
        # 检查是否是文件夹
        if os.path.isdir(folder_path):
            # 检查文件夹名称是否包含'submission'
            if 'submission' in folder_name:
                try:
                    # 获取该文件夹下的所有文件
                    files = os.listdir(folder_path)
                    # 计算文件数量
                    file_count = len([f for f in files if os.path.isfile(os.path.join(folder_path, f))])
                    
                    # 如果文件数量大于等于4，添加到结果列表
                    if file_count >= 4:
                        submission_folders.append(folder_name)
                except PermissionError:
                    # 如果没有访问权限，则跳过
                    continue
    
    # 按照名称中的数字从小到大排序
    submission_folders.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))

    return submission_folders

# 示例用法
path = '/home'  # 请根据实际情况修改路径
result = get_submission_folders(path)

import re

def extract_last_three_numbers(folder_list):
    numbers = []  # 存储提取出的后三位数字
    
    for folder_name in folder_list:
        # 使用正则表达式提取名称中的数字部分
        match = re.search(r'(\d+)_submission', folder_name)
        if match:
            full_number = match.group(1)  # 获取完整的数字
            
            # 提取后三位数字
            last_three_digits = full_number[-3:]  # 取最后三位
            
            # 前补零，确保为三位数
            formatted_number = last_three_digits.zfill(3)
            numbers.append(last_three_digits)  # 转为整数并添加到列表中

    return numbers

# 示例用法
result = extract_last_three_numbers(result)

import pandas as pd

def save_to_excel(data, file_name='submission.xlsx'):
    # 将列表转化为 DataFrame
    df = pd.DataFrame(data, columns=['Numbers'])
    
    # 保存到 Excel 文件
    df.to_excel(file_name, index=False)
    print('done')


        
save_to_excel(result)
