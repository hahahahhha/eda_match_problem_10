from PIL import Image
import os

def resize_images(input_dir, output_dir, target_size):
    """
    批量调整图片大小。
    
    参数:
        input_dir (str): 输入图片文件夹路径。
        output_dir (str): 输出图片文件夹路径。
        target_size (tuple): 目标大小 (width, height)。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有图片
    for img_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.jpg')

        
        try:
            # 打开图片
            with Image.open(input_path) as img:
                # 调整大小并保存
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                # 保存为 JPG 格式
                resized_img.convert("RGB").save(output_path, "JPEG")
                print(f"Resized {img_name} to {target_size} and saved as JPG.")

                os.remove(input_path)
                print(f"Deleted original file: {img_name}")
                
        except Exception as e:
            print(f"Error processing {img_name}: {e}")


# 示例：调整图片为 640x640
main_dir=r'C:\Users\13617\Desktop\mycode\eda_match\datasets\eda\images'
for subset in ['test','train','val']:
    aaapath=os.path.join(main_dir,subset)
    resize_images(
        input_dir=aaapath,
        output_dir=aaapath,
        target_size=(640, 640)
    )
