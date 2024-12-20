import importlib.util
import warnings
import os
import sys 


def dynamic_import(group_id):
    """
    动态导入指定参赛队伍的 my_solution 模块
    """
    group_info = group_code_info[group_id]
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

def handler(signum, frame):
    raise TimeoutError("Operation timed out")

import signal
import tqdm 
import time 
import traceback
def run_tests(solution,group_id,timeout=10*60):
    load_dir = r'/home/public/public/images'
    save_dir = f'/home/public/public/output/{group_id}/generate'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        

    for file in tqdm.tqdm(os.listdir(load_dir)[0:1]):
        print('-----------------------------------------Begin File:', file)
        save_name = file.replace('.png', '.txt')
        txt_save_dir=os.path.join(save_dir, save_name)
        # if os.path.exists(txt_save_dir):
        #     print('[Warnning]-----------------------------------------File exist!!skip!!', file)
        #     continue
        st=time.time()
        signal.signal(signal.SIGALRM,handler)
        signal.alarm(timeout)
        result={"ckt_type":"","ckt_netlist":[]}
        try:
            result = solution(os.path.join(load_dir, file))
            if isinstance(result,str):
                result=eval(result)
            result['pic_ct']=round(time.time()-st,3)
        except TimeoutError:
            result['error']=f'timeout:{timeout}s'
            result['pic_ct']=round(time.time()-st,3)
            print(f'timeout:{timeout}s')
        except:
            result['error']=f'{traceback.format_exc()}'
            result['pic_ct']=round(time.time()-st,3)
            print(f'{traceback.format_exc()}')
        finally:
            signal.alarm(0)
            
        with open(txt_save_dir, 'w') as f:
            f.write(str(result))
        print('-----------------------------------------File completed:', file)

    print(f"******************************************{group_id} finished!!******************************************")

if __name__ == '__main__':
    # 忽略警告信息
    warnings.filterwarnings("ignore", category=FutureWarning)
    ##说明：1. 修改group_code_info的信息，参赛队伍的信息,参赛队的id，code_path:项目代码地址，module_name：模块地址
    ##     2. 结果会在/home/public/public/output/下 ，每个参赛队的id创建一个文件夹    
    group_code_info = {
        # "eda241003": {"code_path": "/home/eda241003_submission/EDAProjectEda241003", "module_name": "my_solution"},
        # "eda241002": {"code_path": "/home/eda241002_submission/submission", "module_name": "my_solution"},
        # "eda241030": {"code_path": "/home/eda241030_submission/codes", "module_name": "my_solution"}, 
        "edatest": {"code_path": "/home/eda241072_submission/submission", "module_name": "my_solution"}
    }


    # 遍历所有参赛队伍，导入并运行测试
    for group_id in group_code_info:
        # try:
        # 动态导入模块
        solution_module = dynamic_import(group_id)
        
        # 获取 solution 函数
        solution = getattr(solution_module, 'solution', None)
        if solution is None:
            raise AttributeError(f"Module {group_id} does not have a 'solution' function.")
        # 运行测试
        run_tests(solution,group_id)
        # except Exception as e:
        #     print(f"Error processing group {group_id}: {e}")