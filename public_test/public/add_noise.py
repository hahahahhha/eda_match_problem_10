import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.04):
    """
    向图像添加随机的椒盐噪声。
    
    :param image: 输入的图像 (PIL.Image 对象)
    :param salt_vs_pepper: 盐与胡椒的比例，默认为0.5
    :param amount: 噪声的比例 (0-1 之间的浮点数)，默认为0.04
    :return: 添加噪声后的图像 (PIL.Image 对象)
    """
    # 将图像转换为numpy数组
    img_array = np.array(image)
    
    # 计算要添加噪声的像素数量
    num_salt = np.ceil(amount * img_array.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * img_array.size * (1. - salt_vs_pepper))

    # 添加盐噪声（白色）
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
    img_array[tuple(coords)] = 1 if img_array.ndim == 2 else 255  # 灰度图或彩色图

    # 添加胡椒噪声（黑色）
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
    img_array[tuple(coords)] = 0

    # 将numpy数组转换回PIL.Image对象
    return Image.fromarray(img_array)

# 加载图像

from  pathlib import Path
import cv2
image_dir=Path("/home/public/cal_score/images_122_20241115")
save_dir=Path("/home/public/cal_score/images_122_20241115_noise")
image_list=image_dir.glob("*.png")
for image_path in image_list:
    print(image_path,image_path.parent,image_path.name,image_path.stem)
    
    # image_path = '/home/libo/work/dataset/check_130val_2_113/001.png'
    image = Image.open(image_path).convert('RGB')  # 确保图像为RGB模式

#     # 添加椒盐噪声
    noisy_image = add_salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.001)  # 5% 的噪声

    # 将PIL.Image对象转换为OpenCV格式（BGR）
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    noisy_image_bgr = cv2.cvtColor(np.array(noisy_image), cv2.COLOR_RGB2BGR)
    print(str(save_dir / f"{image_path.stem}"))
    cv2.imwrite(str(save_dir / f"{image_path.name}"),noisy_image_bgr)
    # cv2.imwrite(str(save_dir / f"{image_path.stem}_noise.png"),noisy_image_bgr)