import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置显示
pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 加载数据 ====================
print("="*80)
print("传统机器学习模型训练与比较")
print("="*80)

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
data_file = os.path.join(desktop_path, "concrete_final_features.csv")

try:
    df = pd.read_csv(data_file)
    print(f"✓ 成功读取特征文件: {data_file}")
    print(f"  数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
except FileNotFoundError:
    print("✗ 未找到文件，尝试在当前目录查找...")
    df = pd.read_csv("concrete_final_features.csv")
    print("✓ 从当前目录读取")

# 分离特征和目标
target = 'Concrete Compressive Strength'
X = df.drop(columns=[target])
y = df[target]

print(f"\n特征数: {X.shape[1]}, 目标变量: {target}")
print("特征列表:", list(X.columns))

# ==================== 2. 划分训练集和测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# ==================== 3. 定义模型和Pipeline ====================
# 对需要标准化的模型，构建包含 StandardScaler 的 Pipeline
models = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    'Ridge Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ]),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'SVR (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR(kernel='rbf', C=1.0, gamma='scale'))
    ]),
    'KNN (k=5)': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', KNeighborsRegressor(n_neighbors=5))
    ])
}

# ==================== 4. 训练和评估 ====================
results = []

print("\n" + "="*80)
print("模型训练与测试集评估")
print("="*80)

for name, model in models.items():
    # 训练
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # 交叉验证（5折）R²（注意Pipeline可直接用于CV）
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results.append({
        'Model': name,
        'R² (Test)': r2,
        'RMSE (Test)': rmse,
        'MAE (Test)': mae,
        'CV R² (mean)': cv_scores.mean(),
        'CV R² (std)': cv_scores.std()
    })
    
    print(f"\n{name}:")
    print(f"  R² (测试集): {r2:.4f}")
    print(f"  RMSE (测试集): {rmse:.2f} MPa")
    print(f"  MAE (测试集): {mae:.2f} MPa")
    print(f"  CV R² (均值±标准差): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 转换为DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R² (Test)', ascending=False)

# ==================== 5. 结果比较可视化 ====================
print("\n" + "="*80)
print("模型性能比较")
print("="*80)
print(results_df[['Model', 'R² (Test)', 'RMSE (Test)', 'MAE (Test)']].round(4))

# 绘制性能对比图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R² 条形图
sns.barplot(data=results_df, x='Model', y='R² (Test)', ax=axes[0], palette='viridis')
axes[0].set_title('R² (测试集)')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=45)

# RMSE 条形图
sns.barplot(data=results_df, x='Model', y='RMSE (Test)', ax=axes[1], palette='magma')
axes[1].set_title('RMSE (MPa)')
axes[1].tick_params(axis='x', rotation=45)

# MAE 条形图
sns.barplot(data=results_df, x='Model', y='MAE (Test)', ax=axes[2], palette='plasma')
axes[2].set_title('MAE (MPa)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
compare_plot = os.path.join(desktop_path, "model_comparison.png")
plt.savefig(compare_plot, dpi=300, bbox_inches='tight')
plt.show()
print(f"\n✓ 性能比较图已保存: {compare_plot}")

# ==================== 6. 保存结果到CSV ====================
result_file = os.path.join(desktop_path, "model_performance.csv")
results_df.to_csv(result_file, index=False)
print(f"✓ 模型性能表已保存: {result_file}")

# ==================== 7. 输出最佳模型 ====================
best_model = results_df.iloc[0]
print("\n" + "="*80)
print("最佳模型推荐")
print("="*80)
print(f"根据 R² 指标，最佳模型为: {best_model['Model']}")
print(f"  R²: {best_model['R² (Test)']:.4f}")
print(f"  RMSE: {best_model['RMSE (Test)']:.2f} MPa")
print(f"  MAE: {best_model['MAE (Test)']:.2f} MPa")

# 对最佳模型进行更详细的评估（可选）
best_estimator = models[best_model['Model']]
if hasattr(best_estimator, 'named_steps'):
    # Pipeline
    model_inside = best_estimator.named_steps['regressor']
    if hasattr(model_inside, 'feature_importances_'):
        importances = model_inside.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        print("\n最佳模型的特征重要性（Top 5）:")
        print(feat_imp.head(5).round(4))

print("\n" + "="*80)
print("模型训练与比较完成！")
print("="*80)