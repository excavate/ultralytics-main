# detect with best.pt

# 1. 命令行检测
# yolo predict model='./runs/detect/train32/weights/best.pt' source='./ultralytics-main/datasets/product_8.5_augment/images/val'

# 2. 脚本检测  最好不用，还得自己定义文件夹以及输出格式
from ultralytics import YOLO
import os

model_path='../runs/detect/train32/weights/best.pt'
if os.path.exists(model_path):
    print("文件夹存在！")
# 加载训练好的模型
model = YOLO(model_path)

# 设置图像文件夹路径和结果保存路径
image_folder = './datasets/scratch/images/test-trained'
result_folder = './test_results'
os.makedirs(result_folder, exist_ok=True)

# 遍历文件夹中的所有图像文件
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 检查文件类型
        img_path = os.path.join(image_folder, filename)
        
        # 执行检测
        results = model(img_path)
        
        # 遍历每个结果对象并保存检测结果图像
        for i, result in enumerate(results):
            result_img_path = os.path.join(result_folder, f'result_{i}_{filename}')
            result.save(result_img_path)
        
            # 获取检测到的对象信息并保存到文本文件
            result_txt_path = os.path.join(result_folder, f'results_{i}_{filename}.txt')
            with open(result_txt_path, 'w') as f:
                for bbox in result.boxes:
                    f.write(f"类别: {bbox.cls}, 置信度: {bbox.conf}, 坐标: {bbox.xyxy}\n")
        
        print(f"Processed {filename}")

print("Detection complete.")