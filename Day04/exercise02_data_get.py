import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo

# 1. 从UCI数据库获取混凝土抗压强度数据集
print("正在从UCI数据库获取Concrete Compressive Strength数据集...")
try:
    # 使用ucimlrepo库获取数据（推荐方法）
    concrete_data = fetch_ucirepo(id=165)  # 数据集的ID为165
    
    # 提取特征和目标变量
    X = concrete_data.data.features
    y = concrete_data.data.targets
    
    # 合并为完整数据集
    df = pd.concat([X, y], axis=1)
    
    print("数据获取成功！")
    
except ImportError:
    print("ucimlrepo库未安装，正在使用备用方法...")
    # 备用方法：直接从UCI网站读取CSV
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    try:
        df = pd.read_excel(url)
        print("通过备用方法获取数据成功！")
    except Exception as e:
        print(f"备用方法也失败了: {e}")
        print("请确保已安装: pip install ucimlrepo openpyxl")
        exit()

# 2. 显示数据基本信息
print("\n" + "="*60)
print("数据基本信息:")
print("="*60)

# 显示数据维度
print(f"\n数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")

# 显示字段名和类型
print("\n字段类型:")
print("-"*40)
print(df.dtypes)

# 3. 显示前五行数据
print("\n前五行数据:")
print("-"*40)
print(df.head())

# 4. 显示更多统计信息
print("\n数据统计描述:")
print("-"*40)
print(df.describe())

# 5. 检查缺失值
print("\n缺失值检查:")
print("-"*40)
print(df.isnull().sum())

# 6. 保存到桌面
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_name = "concrete_compressive_strength.csv"
file_path = os.path.join(desktop_path, file_name)

try:
    df.to_csv(file_path, index=False)
    print(f"\n数据集已成功保存到: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path) / 1024:.2f} KB")
except Exception as e:
    print(f"\n保存文件时出错: {e}")
    # 尝试保存到当前目录作为备选
    alt_path = "./concrete_compressive_strength.csv"
    df.to_csv(alt_path, index=False)
    print(f"已保存到当前目录: {alt_path}")

# 7. 数据集列名说明
print("\n" + "="*60)
print("数据集列名说明:")
print("="*60)
print("""
- Cement (kg/m³): 水泥含量
- Blast Furnace Slag (kg/m³): 矿渣含量
- Fly Ash (kg/m³): 粉煤灰含量
- Water (kg/m³): 水含量
- Superplasticizer (kg/m³): 减水剂含量
- Coarse Aggregate (kg/m³): 粗骨料含量
- Fine Aggregate (kg/m³): 细骨料含量
- Age (day): 养护龄期（天）
- Concrete Compressive Strength (MPa): 混凝土抗压强度（目标变量）
""")

print("\n数据获取完成！可以开始进行机器学习实训了。")
print(f"数据集已准备就绪，共 {df.shape[0]} 个样本，{df.shape[1]-1} 个特征。")
