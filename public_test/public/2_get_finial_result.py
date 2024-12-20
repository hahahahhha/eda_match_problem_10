import time
import warnings
import os
from math import log10  
from utils_ged import *
from pathlib import Path
import json

TIMEOUT = 1


# 单个测例的测试函数
def get_ged_result(generated_netlist:dict,ground_truth_netlist:dict,id):
    # generated_netlist 是一个字典

    try:
        case_id = id # 电路编号
        
        # 获取待测的电路连接
        generated = HeteroGraph()
        ground_truth = HeteroGraph()
        # 将字典形式的网表转化为等价的异构图
        _,ckt_graph,_= generated.generate_all_from_json(generated_netlist)

        _,true_graph,_= ground_truth.generate_all_from_json(ground_truth_netlist) 

        import time

        start_time = time.time()

        test2_result = ged(ckt_graph, true_graph, case_id, timeout = TIMEOUT, task_name = id)
        
        end_time = time.time()

        execution_time = end_time - start_time

    except:
        print(f"get ged bad:{id}",traceback.format_exc())
        test2_result=(999,999)
        execution_time=999
    
    # 生成单个测例的测试报告
    
    report = {'ged_val':test2_result,'ged_ct':execution_time}
    print(f"{id}:",report)
    generated_netlist.update(report)
    return generated_netlist


# 单个测例的测试函数
def get_function_acc(all_result:list):
    # generated_netlist 是一个字典
    for net_info in all_result:
        try:
            import time
            case_id=net_info['case_id']
            generated_netlist=net_info['ckt_netlist']
            ground_truth_netlist=net_info['true_netlist']
            
            start_time = time.time()
            # 获取待测的电路功能
            ckt_label = net_info['ckt_type']
            # 检测电路功能是否判断正确
            test1_result = 1 if ckt_label == ground_truth_netlist['ckt_type'] else 0
            execution_time = round(time.time() - start_time,4)
        except:
            print(f"get function acc:{case_id}",traceback.format_exc())
            test1_result=0
            execution_time=999

        report = {'function_acc':test1_result,'function_ct':execution_time}
        net_info.update(report)
    return all_result

# 格式化输出测试报告
def generate_report(reports,save_dir):
    ##整理格式
    import pandas as pd 
    pd_data=pd.DataFrame.from_dict(reports)[["case_id","function_acc","ged_val","pic_ct"]]
    pd_data=pd_data.rename(columns={"case_id":"测例编号","function_acc":'功能识别精确度F',"ged_val":'网表识别精确度K',"pic_ct":'运行时间'})
    reports=pd_data.to_dict(orient='records')
   
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
    K_total_ct = 0
    num = 0
    for report in reports:
        row = []
        for header in headers:
            row.append(str(report[header]))
        rows.append(" | ".join(row))
        F_total += report['功能识别精确度F']
        K_total_approx += report['网表识别精确度K'][0]
        K_total_exact += report['网表识别精确度K'][1]
        K_total_ct += report['运行时间']
        num +=1

        
    final_F = F_total / num
    final_K_approx = 1 / log10(10 + K_total_approx)
    final_K_exact = 1 / log10(10 + K_total_exact)
    final_ct=K_total_ct / num


    markdown_table = f"# 测试报告\n\n{header_line}\n{separator_line}\n" + "\n".join(rows)
    final_result = f"\n ## 综合测试结果\n\n总得分| 功能识别精确度F | 网表识别精确度K | 平均运行时间 \n--- | --- | --- | --- |  \n 略 | {final_F:.4f} | {final_K_approx:.4f}, {final_K_exact:.4f} | {final_ct:.4f}  \n\n"
    markdown_table += final_result
    report_path = os.path.join(save_dir,'validation_report.md')
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(markdown_table)
    print(f"测试报告已保存在 {report_path}")


import ast
def read_dict_from_file(file_path,encoding='utf-8'):
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
    # parent_directory = os.path.dirname(file1_directory)
    parent_directory="/home/nctieda10/public/"
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

import traceback
def process_file(file_path):
    true_netlist_path,id = find_true(file_path)
    true_netlist = read_dict_from_file(true_netlist_path)
    try:
        
        generated_netlist = read_dict_from_file(file_path)
        ##############测试部分###################

        report = get_ged_result(generated_netlist,true_netlist,id)
    except :
        print(f"测例 {id} :",traceback.format_exc())
        report={'ged_val':999,'ged_ct':999}

    ########################################
    generated_netlist['case_id']=id
    generated_netlist['true_netlist']={}
    generated_netlist.update(report)
    print(f'测例 {id} 已完成')
    return generated_netlist
    
# 运行所有测例的函数
import json 
def run_tests(dir_path:str,save_dir:str,mode:int):# 输入含有生成网表的文件夹路径 dir_path
    reports = []

    ged_txt_dir=os.path.join(save_dir,"ged_result.txt")

    if  os.path.exists(ged_txt_dir):
        reports=read_dict_from_file(ged_txt_dir)

    if (mode==0 or mode==2) and len(reports)==0:#get ged
        from dask.distributed import Client,LocalCluster
        cluster = LocalCluster(threads_per_worker=1, n_workers=10)
        # 等待几秒钟让集群启动
        while not cluster.scheduler_address:
            time.sleep(0.1)
        
        cluster.adapt(minimum=4, maximum=40) 
        client = Client(cluster)
        client.wait_for_workers(4)
        futures = []
        input_params = get_file_paths(dir_path)

        for parameters in input_params:
            future = client.submit(process_file, parameters)
            futures.append(future)
        reports = client.gather(futures)
        
        cluster.close()
        client.close()
        with open(ged_txt_dir, 'w') as f:
            print("正在写入ged result:",ged_txt_dir)
            reports_str=json.dumps(reports,indent=4,ensure_ascii=False)
            f.write(reports_str)
    
    if mode==1 or mode==2:
        # 格式化并输出测试结果  
        get_function_acc(reports)
        all_info_txt=os.path.join(save_dir,"all_info.txt")
        with open(all_info_txt, 'w') as f:
            reports_str=json.dumps(reports,indent=4,ensure_ascii=False)
            f.write(reports_str)
        generate_report(reports,save_dir)
        
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse input arguments")
    parser.add_argument('--mode', type=int, help='only ged:0,only get circuit acc:1(you must get ged then get acc),all:2')
    parser.add_argument('--output_file', default='./result_test/',type=str, help='Path to the output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 忽略警告信息
    warnings.filterwarnings("ignore", category=FutureWarning)
    args=parse_arguments()
    # dir_path: 参赛队伍需将其修改为该队生成网表的文件路径
    
    result_list=[
        # "/home/public/public/output/eda241030",
        # "/home/eda241035_submission/submission",
        # "/home/eda241041_submission/submission",
        # "/home/eda241063_submission/submission/software",
        # "/home/eda241055_submission/submission/codes",
        # "/home/eda241018_submission/codes",
        # "/home/eda241002_submission/submission",
        "/home/public/public/output/eda241038",
    ]

    save_root=args.output_file
    for file_path in result_list:
        dir_path=os.path.join(file_path,"generate")

        save_dir=os.path.join(save_root,file_path.split("/")[-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        run_tests(dir_path,save_dir,args.mode)
        print(f"{file_path} deal completed!")