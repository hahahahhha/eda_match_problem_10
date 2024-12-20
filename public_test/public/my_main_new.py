import sys 
import os 
os.chdir("/home/eda241030_submission/codes")
sys.path.append('/home/eda241030_submission/codes')
from my_solution import solution

import numpy as np
import cv2


def get_image(image_path):
   
    with open(image_path, 'rb') as img_file:
        case_image = img_file.read()
    return case_image


if __name__ == '__main__':
    import os

    # load_dir = r'C:\Users\PC\Desktop\public\images'
    # save_dir = r'C:\Users\PC\Desktop\public\generate'

    print("llll:",sys.path)

    load_dir = r'/home/public/public/images'
    save_dir = r'./generate_new'
    save_dir_ct = r'./generate_ct'
    if not os.path.exists(save_dir_ct):
        os.mkdir(save_dir_ct)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    import tqdm 
    import time 
    for file in tqdm.tqdm(os.listdir(load_dir)):
        print('Begin File:', file)
        st=time.time()
        
        result = solution(get_image(os.path.join(load_dir, file)))
        
        # result = solution(os.path.join(load_dir, file))

        ct=round(time.time()-st,3)
        
        save_name = file.replace('.png', '.txt')
        with open(os.path.join(save_dir, save_name), 'w') as f:
            f.write(str(result))
        save_name=file.replace('.png', f'_{ct}.txt')
        with open(os.path.join(save_dir_ct, save_name), 'w') as f:
            f.write(str(result))

    print("finished!!")
