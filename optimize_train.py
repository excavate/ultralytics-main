import os
from itertools import product
from ultralytics import YOLO

# 定义超参数搜索空间
param_grid = {
    'lr0': [0.001, 0.01, 0.1],
    'batch': [8, 16, 32],
    'epochs': [100]
}

# 生成超参数的所有组合
param_combinations = list(product(param_grid['lr0'], param_grid['batch'], param_grid['epochs']))

# 定义数据集路径
data_yaml = r'D:\program\object detection\ultralytics-main\datasets\product_8.7_augment\8.7_laptop.v2i.yolov8\data.yaml'

# 记录结果
results = []

# 遍历所有超参数组合
for (lr0, batch, epochs) in param_combinations:
    print(f"Training with lr0={lr0}, batch={batch}, epochs={epochs}")
    
    # 加载模型
    model = YOLO("yolov8n.pt")
    
    # 训练模型
    result=model.train(data=data_yaml, epochs=epochs, lr0=lr0, batch=batch,workers=0)
    result_dict=result.results_dict
    
    # 记录结果
    results.append({
        'lr0': lr0,
        'batch': batch,
        'epochs': epochs,
        'Precision':result_dict['metrics/precision(B)'],
        'Recall':result_dict['metrics/recall(B)'],
        'mAP50': result_dict['metrics/mAP50(B)'],
        'mAP50-95': result_dict['metrics/mAP50-95(B)']
    })

# 保存结果到csv
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)

# 绘制不同lr0和batch参数组合的mapp50和mAP50-95的柱状图
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.barplot(x='lr0', y='mAP50', hue='batch', data=df)
plt.title('mAP50 by lr0 and batch')
plt.show()


# 找到最佳超参数组合
best_params = max(results, key=lambda x: x['mAP50'])
print(f"Best parameters based on mAP50: {best_params}")

# 找到mAP50-95最高的超参数组合
best_params_map50_95 = max(results, key=lambda x: x['mAP50-95'])
print(f"Best parameters based on mAP50-95: {best_params_map50_95}")
