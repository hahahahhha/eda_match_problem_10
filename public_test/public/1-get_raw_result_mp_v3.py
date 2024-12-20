from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import warnings
import os
import sys 
import concurrent


def dynamic_import(group_id):
    """
    动态导入指定参赛队伍的 my_solution 模块
    """
    group_info = group_code_info_[group_id]
    code_path = group_info['code_path']
    module_name = group_info['module_name']
    
    # 构建模块的完整路径
    # 将模块所在目录添加到 sys.path
    sys.path.append(code_path)
    os.chdir(code_path)
    module_path = os.path.join(code_path, f"{module_name}.py")
    
    # 动态导入模块
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module

def get_image(image_path):
    with open(image_path,'rb') as img_file:
        case_image = img_file.read()
    return case_image



import signal
import tqdm 
import time 
import traceback

def run_test(solution, filepath, txt_save_dir):

    print('-----------------------------------------Begin File:', filepath)
    result={"ckt_type":"","ckt_netlist":[],"pic_ct":0,"error":[]}
    st=time.time()
    
    try:
        result = solution(filepath)
        if isinstance(result,str):
            result=eval(result)
        result['error']=[]
    except:
        result['error'].append(f'{traceback.format_exc()}')
        print(f'{traceback.format_exc()}')

    result['pic_ct']=round(time.time()-st,3)
    print('-----------------------------------------end File:', filepath,',error:',result['error'])
    return result

def run_tests(solution, group_id, timeout=60):
    load_dir = '/home/public/cal_score/images_122_20241115_noise'
    save_dir = f'/home/public/cal_score/output_20241115_tmp/{group_id}/generate'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    max_threads = 8  # Set the desired number of threads

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {}
        file_list=[file for file in os.listdir(load_dir) if "resize" not in file]

        for filename in tqdm.tqdm(file_list):
            filepath = os.path.join(load_dir, filename)
            save_name = os.path.basename(filepath).replace('.png', '.txt')
            txt_save_dir = os.path.join(save_dir, save_name)
            if os.path.exists(txt_save_dir):
                print('[Warnning]-----------------------------------------File exist!!skip!!', filepath)
                continue

            if "resize" in os.path.basename(filepath):
                continue

            future = executor.submit(run_test, solution, filepath, txt_save_dir)
            futures[future] = filename

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            filename = futures[future]
            result={"ckt_type":"","ckt_netlist":[],"pic_ct":999,"error":[]}
            try:
                # 捕捉超时异常
                result=future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                result['error'].append((f"TimeoutError: File {filename} exceeded {timeout} seconds."))
                print(f"TimeoutError: File {filename} exceeded {timeout} seconds.")
            except Exception as e:
                result['error'].append((f"Error processing file {filename}: {traceback.format_exc()}"))
                print(f"Error processing file {filename}: {traceback.format_exc()}")
            
            save_name = filename.replace('.png', '.txt')
            txt_save_dir = os.path.join(save_dir, save_name)
            with open(txt_save_dir, 'w') as f:
                f.write(str(result))
                print('-----------------------------------------File completed:', save_name)

if __name__ == '__main__':
    # 忽略警告信息
    warnings.filterwarnings("ignore", category=FutureWarning)
    ##说明：1. 修改group_code_info的信息，参赛队伍的信息,参赛队的id，code_path:项目代码地址，module_name：模块地址
    ##     2. 结果会在/home/public/public/output/下 ，每个参赛队的id创建一个文件夹    
    group_code_info_ = {
        "eda241002": {"code_path": "/home/eda241002_submission/submission", "module_name": "my_solution"},
        "eda241003": {"code_path": "/home/eda241003_submission/submission/submit_241003/EDAProject", "module_name": "my_solution"},
        "eda241007": {"code_path": "/home/eda241007_submission/submission", "module_name": "my_solution"},
        # "eda241009": {"code_path": "/home/eda241009_submission/EDA_cat_1113", "module_name": "solution"},
        "eda241011": {"code_path": "/home/eda241011_submission/submission", "module_name": "my_solution"},
        # "eda241013": {"code_path": "/home/eda241013_submission/submission", "module_name": "my_solution"},
        # "eda241014": {"code_path": "/home/eda241014_submission/submission", "module_name": "netlist"},
        # "eda241017": {"code_path": "/home/eda241017_submission", "module_name": "my_solution"},
        "eda241018": {"code_path": "/home/eda241018_submission/submission", "module_name": "my_solution"},
        "eda241021": {"code_path": "/home/eda241021_submission/submission", "module_name": "my_solution"},
        "eda241022": {"code_path": "/home/eda241022_submission/submission", "module_name": "my_solution"},
        "eda241023": {"code_path": "/home/eda241023_submission/submission", "module_name": "my_solution"},
        "eda241028": {"code_path": "/home/eda241028_submission/submission", "module_name": "my_solution"},
        "eda241030": {"code_path": "/home/eda241031_submission/submission", "module_name": "my_solution"},
        # "eda241031": {"code_path": "/home/eda241031_submission/submission", "module_name": "my_solution"}
        # "eda241033": {"code_path": "/home/eda241033_submission/submission/submit_example", "module_name": "my_solution"},
        "eda241035": {"code_path": "/home/eda241035_submission/submission", "module_name": "my_solution"},
        "eda241036": {"code_path": "/home/eda241036_submission/submission", "module_name": "my_solution"},
        "eda241037": {"code_path": "/home/eda241037_submission/submission", "module_name": "my_solution"},
        "eda241039": {"code_path": "/home/eda241039_submission/submission", "module_name": "my_solution"},
        "eda241040": {"code_path": "/home/eda241040_submission/eda241040_submission", "module_name": "my_solution"},
        # "eda241041": {"code_path": "/home/eda241041_submission/submission", "module_name": "solution"},
        "eda241042": {"code_path": "/home/eda241042_submission/submission", "module_name": "my_solution"},
        # "eda241044": {"code_path": "/home/eda241044_submission/submission/yolov10-main", "module_name": "my_solution"},
        # "eda241045": {"code_path": "/home/eda241045_submission/submission", "module_name": "main"},
        "eda241046": {"code_path": "/home/eda241046_submission", "module_name": "my_solution"},
        # "eda241047": {"code_path": "/home/eda241047_submission/submission", "module_name": "my_solution"},
        "eda241048": {"code_path": "/home/eda241048_submission", "module_name": "my_solution"},
        # "eda241050": {"code_path": "/home/eda241050_submission/submission", "module_name": "main"},
        "eda241055": {"code_path": "/home/eda241055_submission/submission/codes20241113", "module_name": "my_solution"},
        # "eda241058": {"code_path": "/home/eda241058_submission/submission", "module_name": "main"},
        "eda241062": {"code_path": "/home/eda241062_submission/submission_old/yolov5_7", "module_name": "mysoln"},
        "eda241063": {"code_path": "/home/eda241063_submission/submission/software", "module_name": "my_solution"},
        "eda241068": {"code_path": "/home/eda241068_submission/submission", "module_name": "my_solution"},
        # "eda241072": {"code_path": "/home/eda241072_submission/submission", "module_name": "my_solution"},
    }


    # 遍历所有参赛队伍，导入并运行测试
    for group_id in group_code_info_:
        print(f"*****************************start {group_id} libo******************************")
        try:
            # 动态导入模块
                
            solution_module = dynamic_import(group_id)
            
            # 获取 solution 函数
            solution = getattr(solution_module, 'solution', None)
            if solution is None:
                raise AttributeError(f"Module {group_id} does not have a 'solution' function.")
            # 运行测试
            run_tests(solution,group_id)
        except Exception as e:
            print(f"Error processing group {group_id}: {e}")
        print(f"******************************************{group_id} finished!!******************************************")
        

