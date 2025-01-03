import time
import warnings
import os
from math import log10  
from utils import *


# 单个测例的测试函数
def test_case(generated_netlist,ground_truth_netlist,id,):
    case_id = id # 电路编号
    
    # 获取待测的电路连接
    generated = HeteroGraph()
    ground_truth = HeteroGraph()
    # 将字典形式的网表转化为等价的异构图
    _,ckt_graph,_= generated.generate_all_from_json(generated_netlist)

    _,true_graph,_= ground_truth.generate_all_from_json(ground_truth_netlist) 

    import time

    start_time = time.time()

    test2_result = ged(ckt_graph, true_graph,case_id)
    
    end_time = time.time()

    execution_time = end_time - start_time


    # 获取待测的电路功能
    ckt_label = generated_netlist['ckt_type']
    # 检测电路功能是否判断正确
    test1_result = 1 if ckt_label == ground_truth_netlist['ckt_type'] else 0
    
    # 生成单个测例的测试报告
    report = {'测例编号':case_id,'功能识别精确度F':test1_result,'网表识别精确度K':test2_result,'运行时间':execution_time}
    return report

# 格式化输出测试报告
def generate_report(reports):
   
    def extract_case_number(report):
        id = report['测例编号']
        return int(id)
    
    reports.sort(key=extract_case_number)
   
    # 设置表头
    headers = ["测例编号",'功能识别精确度F','网表识别精确度K','运行时间']

    # 创建表头行和分隔行
    header_line = " | ".join(headers)
    separator_line = " | ".join(["---"] * len(headers))

    # 创建数据行
    rows = []
    F_total = 0
    K_total_approx = 0
    K_total_exact = 0
    num = 0
    for report in reports:
        row = []
        for header in headers:
            row.append(str(report[header]))
        rows.append(" | ".join(row))
        F_total += report['功能识别精确度F']
        K_total_approx += report['网表识别精确度K'][0]
        K_total_exact += report['网表识别精确度K'][1]
        num +=1

        
    final_F = F_total / num
    final_K_approx = 1 / log10(10 + K_total_approx)
    final_K_exact = 1 / log10(10 + K_total_exact)


    markdown_table = f"# 测试报告\n\n{header_line}\n{separator_line}\n" + "\n".join(rows)
    final_result = f"\n ## 综合测试结果\n\n总得分| 功能识别精确度F | 网表识别精确度K \n--- | --- | ---  \n 略 | {final_F:.4f} | {final_K_approx:.4f}, {final_K_exact:.4f}   \n\n"
    markdown_table += final_result
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'validation_report.md')
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(markdown_table)
    print(f"测试报告已保存在 {report_path}")


import ast
def read_dict_from_file(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        dict_str = file.read()
    # 使用 ast.literal_eval 将字符串转换为字典
    dictionary = ast.literal_eval(dict_str)
    return dictionary


def find_true(file1_path):
    # 获取文件1所在的文件夹路径
    file1_directory = os.path.dirname(file1_path)
    # 获取文件1的文件名（包括扩展名）
    file1_name_with_ext = os.path.basename(file1_path)

    # 获取文件1所在文件夹的父文件夹路径
    parent_directory = os.path.dirname(file1_directory)
    # 构造“true”文件夹的绝对路径
    true_directory_path = os.path.join(parent_directory, 'true')
    # 构造文件2的绝对路径
    file2_path = os.path.join(true_directory_path, file1_name_with_ext)
    # 检查“true”文件夹是否存在，以及文件2是否存在
    if os.path.isdir(true_directory_path):
        if os.path.isfile(file2_path):
            return file2_path,file1_name_with_ext[:-4]
        else:
            print(f"文件2 '{file1_name_with_ext}' 不存在于 '{true_directory_path}' 文件夹中。")
            return None
    else:
        print(f"文件夹 '{true_directory_path}' 不存在。")
        return None


import os

def get_file_paths(folder_path):
    """
    读取指定文件夹路径，返回该文件夹下所有文件的绝对路径列表。

    参数:
    folder_path (str): 要读取的文件夹路径

    返回:
    list: 包含所有文件绝对路径的列表
    """
    file_paths = []
    
    # 确保文件夹路径存在
    if not os.path.isdir(folder_path):
        print(f"错误：'{folder_path}' 不是一个有效的文件夹路径。")
        return file_paths

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 构建绝对路径并添加到列表
            file_path = os.path.abspath(os.path.join(root, file))
            file_paths.append(file_path)

    return file_paths


def process_file(file_path):
    
    generated_netlist = read_dict_from_file(file_path)

    true_netlist_path,id = find_true(file_path)
    true_netlist = read_dict_from_file(true_netlist_path)

    ##############测试部分###################


    report = test_case(generated_netlist,true_netlist,id)
    
    ########################################

    print(f'测例 {report["测例编号"]} 已完成')
    return report
    
# 运行所有测例的函数
def run_tests(dir_path):# 输入含有生成网表的文件夹路径 dir_path
    reports = []

    from dask.distributed import Client,LocalCluster
    cluster = LocalCluster(threads_per_worker=1, n_workers=10)
    cluster.adapt(minimum=4, maximum=40) 
    client = Client(cluster)
    

    
    input_params = get_file_paths(dir_path)
    futures = []
    
    for parameters in input_params:
        future = client.submit(process_file, parameters)
        futures.append(future)
        
    reports = client.gather(futures)
    # 格式化并输出测试结果  
    generate_report(reports)
    
    cluster.close()
    client.close()


if __name__ == '__main__':
    # 忽略警告信息
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # dir_path: 参赛队伍需将其修改为该队生成网表的文件路径
    dir_path = "/home/public/public/generate" # 此处为示例

    run_tests(dir_path)
    







    
    

    
    
