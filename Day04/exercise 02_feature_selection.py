import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# 设置显示
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ==================== 1. 读取清洗后的数据 ====================
print("="*80)
print("特征工程与特征选取")
print("="*80)

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
clean_file = os.path.join(desktop_path, "concrete_clean_data.csv")

if os.path.exists(clean_file):
    df = pd.read_csv(clean_file)
    print(f"✓ 读取清洗后数据: {clean_file}")
    print(f"  原始维度: {df.shape}")
else:
    # 如果没找到，尝试当前目录
    df = pd.read_csv("concrete_clean_data.csv")
    print("✓ 从当前目录读取清洗后数据")

# 保存目标变量和原始特征（以备后续使用）
target = 'Concrete Compressive Strength'
X_raw = df.drop(columns=[target])
y = df[target]

print(f"  特征数: {X_raw.shape[1]}, 样本数: {X_raw.shape[0]}")

# ==================== 2. 构造新特征 ====================
print("\n" + "="*80)
print("2. 构造新特征（基于领域知识）")
print("="*80)

X_eng = X_raw.copy()

# 2.1 胶凝材料总量
X_eng['Binder'] = X_eng['Cement'] + X_eng['Blast Furnace Slag'] + X_eng['Fly Ash']

# 2.2 水胶比 (W/B) —— 更合理的强度指标
X_eng['Water_to_Binder'] = X_eng['Water'] / X_eng['Binder']

# 2.3 水灰比 (W/C) —— 传统指标
X_eng['Water_to_Cement'] = X_eng['Water'] / X_eng['Cement']

# 2.4 矿物掺合料比例
X_eng['Slag_Ratio'] = X_eng['Blast Furnace Slag'] / X_eng['Binder']
X_eng['FlyAsh_Ratio'] = X_eng['Fly Ash'] / X_eng['Binder']

# 2.5 龄期对数变换 (因为分布严重右偏)
X_eng['Log_Age'] = np.log1p(X_eng['Age'])  # log(1+Age)

# 2.6 龄期的平方项 (捕捉非线性)
X_eng['Age_Squared'] = X_eng['Age'] ** 2

# 2.7 减水剂是否使用的二元标识 (很多样本为0)
X_eng['Has_Superplasticizer'] = (X_eng['Superplasticizer'] > 0).astype(int)

# 2.8 粗骨料与细骨料的比例 (影响级配)
X_eng['Aggregate_Ratio'] = X_eng['Coarse Aggregate'] / X_eng['Fine Aggregate']

# 2.9 交互特征: 水泥×龄期 (年龄越大，水泥水化更充分)
X_eng['Cement_Age'] = X_eng['Cement'] * X_eng['Log_Age']

# 2.10 交互特征: 减水剂×水 (减水剂可以减少用水量)
X_eng['Superplasticizer_Water'] = X_eng['Superplasticizer'] * X_eng['Water']

print(f"✓ 新特征构造完成，现有特征数: {X_eng.shape[1]}")

# 查看新特征的描述统计
print("\n新特征统计描述:")
print(X_eng[['Binder', 'Water_to_Binder', 'Water_to_Cement', 'Log_Age']].describe().round(3))

# ==================== 3. 特征选择 ====================
print("\n" + "="*80)
print("3. 特征选择")
print("="*80)

# 3.1 删除方差极低的特征 (几乎不变)
print("\n3.1 方差阈值过滤:")
selector_var = VarianceThreshold(threshold=0.01)
X_var = selector_var.fit_transform(X_eng)
selected_var = X_eng.columns[selector_var.get_support()]
removed_var = X_eng.columns[~selector_var.get_support()]
print(f"  保留特征数: {len(selected_var)}, 删除特征数: {len(removed_var)}")
if len(removed_var) > 0:
    print(f"  删除的特征: {list(removed_var)}")
X_sel = X_eng[selected_var]

# 3.2 基于相关系数的过滤 (与目标变量的相关性)
print("\n3.2 相关性过滤 (保留与目标变量|corr|>0.05):")
correlations = X_sel.corrwith(y).abs()
keep_corr = correlations[correlations > 0.05].index
removed_corr = correlations[correlations <= 0.05].index
print(f"  保留特征数: {len(keep_corr)}, 删除特征数: {len(removed_corr)}")
if len(removed_corr) > 0:
    print(f"  删除的特征: {list(removed_corr)}")
X_sel = X_sel[keep_corr]

# 3.3 基于随机森林特征重要性
print("\n3.3 随机森林特征重要性 (选取Top 10):")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_sel, y)
importances = pd.Series(rf.feature_importances_, index=X_sel.columns).sort_values(ascending=False)
print("特征重要性排名 (Top 10):")
print(importances.head(10).round(4))

# 选择重要性排名前10的特征
top_n = 10
selected_rf = importances.head(top_n).index
X_rf = X_sel[selected_rf]
print(f"\n✓ 最终选取特征数: {len(selected_rf)}")
print(f"  选取的特征: {list(selected_rf)}")

# ==================== 4. 降维 (PCA) - 可选 ====================
print("\n" + "="*80)
print("4. 降维处理 (主成分分析 PCA) - 可选")
print("="*80)

# 标准化数据 (PCA需要标准化)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rf)

# 执行PCA，解释方差达到95%
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
n_pca = X_pca.shape[1]
explained_var = pca.explained_variance_ratio_.sum()

print(f"  原始特征数: {X_rf.shape[1]}")
print(f"  降维后主成分数: {n_pca}")
print(f"  累计解释方差: {explained_var:.2%}")

# 观察主成分载荷
components_df = pd.DataFrame(
    pca.components_,
    columns=X_rf.columns,
    index=[f"PC{i+1}" for i in range(n_pca)]
)
print("\n前3个主成分的载荷:")
print(components_df.head(3).round(3))

# 询问用户是否使用PCA结果
use_pca = input("\n是否使用PCA降维后的数据？(y/n, 默认n): ").strip().lower()
if use_pca == 'y':
    X_final = X_pca
    feature_names = [f"PC{i+1}" for i in range(n_pca)]
    print("✓ 将使用PCA降维后的数据")
else:
    X_final = X_rf.values
    feature_names = X_rf.columns.tolist()
    print("✓ 将使用特征选择后的原始特征")

# ==================== 5. 保存最终数据 ====================
print("\n" + "="*80)
print("5. 保存最终特征矩阵")
print("="*80)

# 构建最终DataFrame
df_final = pd.DataFrame(X_final, columns=feature_names)
df_final[target] = y.values  # 添加目标变量

# 保存到桌面
final_file = os.path.join(desktop_path, "concrete_final_features.csv")
df_final.to_csv(final_file, index=False)
print(f"✓ 最终数据已保存到: {final_file}")
print(f"  最终维度: {df_final.shape[0]} 行 × {df_final.shape[1]} 列")

# 额外保存特征选择信息
feature_info = pd.DataFrame({
    '原始特征': X_raw.columns,
    '是否保留': X_raw.columns.isin(X_rf.columns),
    '重要性得分': rf.feature_importances_ if len(X_rf.columns)==len(X_sel.columns) else 'N/A'
})
info_file = os.path.join(desktop_path, "feature_selection_info.csv")
feature_info.to_csv(info_file, index=False)
print(f"✓ 特征选择信息已保存到: {info_file}")

# ==================== 6. 可视化特征重要性 ====================
print("\n绘制特征重要性条形图...")
plt.figure(figsize=(12, 6))
importances_selected = importances[selected_rf].sort_values()
plt.barh(importances_selected.index, importances_selected.values, color='steelblue')
plt.xlabel('重要性得分')
plt.title('随机森林特征重要性 (Top 10)')
plt.tight_layout()
imp_plot = os.path.join(desktop_path, "feature_importance.png")
plt.savefig(imp_plot, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ 特征重要性图已保存: {imp_plot}")

print("\n" + "="*80)
print("特征工程与特征选择完成！")
print("现在可以使用 'concrete_final_features.csv' 进行机器学习建模。")
print("="*80)