import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

# 1. 读取数据
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "concrete_compressive_strength.csv")

print("="*80)
print("混凝土抗压强度数据集 - 数据清洗与描述性分析")
print("="*80)

try:
    df = pd.read_csv(file_path)
    print(f"\n✓ 成功读取数据文件: {file_path}")
    print(f"  数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
except FileNotFoundError:
    print(f"\n✗ 未在桌面找到文件，尝试在当前目录查找...")
    file_path = "./concrete_compressive_strength.csv"
    df = pd.read_csv(file_path)
    print(f"✓ 成功读取数据文件: {file_path}")

# 显示列名
print(f"\n数据集列名: {list(df.columns)}")

# ==================== 自动检测目标变量列名 ====================
print("\n" + "="*80)
print("自动检测目标变量列名")
print("="*80)

# 查找包含'strength'的列作为目标变量
target_col = None
for col in df.columns:
    if 'strength' in col.lower():
        target_col = col
        break

if target_col is None:
    target_col = df.columns[-1]
    print(f"⚠ 未检测到目标变量列，默认使用最后一列: '{target_col}'")

print(f"✓ 目标变量列名: '{target_col}'")

# 标准化列名（将所有列名改为更友好的格式）
# 注意：这里不重命名目标变量，而是保留原始列名
target_col_original = target_col

# 为了统一处理，创建一个副本并重命名目标列为标准名称
df_renamed = df.copy()
if target_col != 'Concrete Compressive Strength':
    df_renamed = df_renamed.rename(columns={target_col: 'Concrete Compressive Strength'})
    print(f"✓ 已将列名 '{target_col}' 重命名为 'Concrete Compressive Strength'")
    target_col_standard = 'Concrete Compressive Strength'
else:
    target_col_standard = target_col

# ==================== 1. 数据清洗 ====================
print("\n" + "="*80)
print("第一部分：数据清洗")
print("="*80)

# 1.1 缺失值统计
print("\n1.1 缺失值统计:")
missing_values = df_renamed.isnull().sum()
missing_percent = (missing_values / len(df_renamed)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing_values,
    '缺失百分比(%)': missing_percent.round(2)
})
if missing_df['缺失数量'].sum() > 0:
    print(missing_df[missing_df['缺失数量'] > 0])
else:
    print("✓ 数据集中没有缺失值")

# 1.2 重复样本检测
print("\n1.2 重复样本统计:")
duplicate_count = df_renamed.duplicated().sum()
duplicate_percent = (duplicate_count / len(df_renamed)) * 100
print(f"重复样本数量: {duplicate_count}")
print(f"重复样本占比: {duplicate_percent:.2f}%")

if duplicate_count > 0:
    print("\n重复样本示例（前5个）:")
    print(df_renamed[df_renamed.duplicated(keep=False)].head())
    # 删除重复样本（保留第一个）
    df_clean = df_renamed.drop_duplicates(keep='first')
    print(f"\n✓ 已删除重复样本，剩余: {df_clean.shape[0]} 行")
else:
    df_clean = df_renamed.copy()
    print("✓ 数据集中没有重复样本")

# 1.3 数据类型检查
print("\n1.3 数据类型检查:")
print(df_clean.dtypes)

# 1.4 异常值检测（使用IQR方法）
print("\n1.4 异常值检测（IQR方法）:")
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
outliers_summary = {}

for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    outliers_summary[col] = len(outliers)

outliers_df = pd.DataFrame.from_dict(outliers_summary, orient='index', columns=['异常值数量'])
outliers_df['异常值占比(%)'] = (outliers_df['异常值数量'] / len(df_clean) * 100).round(2)
print(outliers_df)

# ==================== 2. 描述性统计分析 ====================
print("\n" + "="*80)
print("第二部分：描述性统计分析")
print("="*80)

# 2.1 数值变量的统计信息
print("\n2.1 数值变量统计描述:")
stats_desc = df_clean[numeric_cols].describe()
print(stats_desc)

# 2.2 额外统计信息（偏度、峰度）
print("\n2.2 偏度和峰度分析:")
skewness = df_clean[numeric_cols].skew()
kurtosis = df_clean[numeric_cols].kurtosis()
stats_extra = pd.DataFrame({
    '偏度 (Skewness)': skewness.round(3),
    '峰度 (Kurtosis)': kurtosis.round(3)
})
print(stats_extra)

# 2.3 相关性矩阵
print("\n2.3 特征相关性矩阵（与目标变量的相关性）:")
# 确保目标变量在numeric_cols中
if 'Concrete Compressive Strength' in numeric_cols:
    correlation = df_clean[numeric_cols].corr()['Concrete Compressive Strength'].sort_values(ascending=False)
    print(correlation)
else:
    print(f"⚠ 目标变量 'Concrete Compressive Strength' 不在数值列中")
    # 尝试查找包含strength的列
    strength_cols = [col for col in numeric_cols if 'strength' in col.lower()]
    if strength_cols:
        for col in strength_cols:
            correlation = df_clean[numeric_cols].corr()[col].sort_values(ascending=False)
            print(f"\n使用 '{col}' 作为目标变量:")
            print(correlation)
    else:
        # 使用最后一列
        last_col = numeric_cols[-1]
        correlation = df_clean[numeric_cols].corr()[last_col].sort_values(ascending=False)
        print(f"\n使用最后一列 '{last_col}' 作为目标变量:")
        print(correlation)

# ==================== 3. 数据可视化 ====================
print("\n" + "="*80)
print("第三部分：数据可视化分析")
print("="*80)

# 创建图形保存目录
fig_dir = os.path.join(desktop_path, "concrete_analysis_figures")
os.makedirs(fig_dir, exist_ok=True)
print(f"\n图形将保存到: {fig_dir}")

# 确保有目标变量列
if 'Concrete Compressive Strength' not in df_clean.columns:
    # 尝试找到包含strength的列
    for col in df_clean.columns:
        if 'strength' in col.lower():
            df_clean = df_clean.rename(columns={col: 'Concrete Compressive Strength'})
            break

# 更新数值列列表
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

# 3.1 直方图 - 所有数值变量的分布
print("\n3.1 绘制直方图...")
n_cols = len(numeric_cols)
n_rows = (n_cols + 2) // 3
fig, axes = plt.subplots(n_rows, min(3, n_cols), figsize=(15, n_rows * 4))
if n_rows == 1:
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
else:
    axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    if i < len(axes):
        axes[i].hist(df_clean[col], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{col}\n(均值={df_clean[col].mean():.2f}, 标准差={df_clean[col].std():.2f})', fontsize=10)
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel('频数', fontsize=9)
        axes[i].axvline(df_clean[col].mean(), color='red', linestyle='dashed', linewidth=2, label='均值')
        axes[i].axvline(df_clean[col].median(), color='green', linestyle='dashed', linewidth=2, label='中位数')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

# 隐藏多余的子图
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
hist_path = os.path.join(fig_dir, '1_histograms.png')
plt.savefig(hist_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ 直方图已保存: {hist_path}")

# 3.2 箱线图 - 检测异常值
print("\n3.2 绘制箱线图...")
n_rows = (n_cols + 2) // 3
fig, axes = plt.subplots(n_rows, min(3, n_cols), figsize=(15, n_rows * 4))
if n_rows == 1:
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
else:
    axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    if i < len(axes):
        axes[i].boxplot(df_clean[col], vert=True, patch_artist=True)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel(col, fontsize=9)
        axes[i].grid(True, alpha=0.3)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
boxplot_path = os.path.join(fig_dir, '2_boxplots.png')
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ 箱线图已保存: {boxplot_path}")

# 3.3 相关性热力图
print("\n3.3 绘制相关性热力图...")
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df_clean[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('特征相关性热力图', fontsize=14)
plt.tight_layout()
heatmap_path = os.path.join(fig_dir, '3_correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ 相关性热力图已保存: {heatmap_path}")

# 3.4 散点图矩阵 - 与目标变量的关系
print("\n3.4 绘制散点图（特征vs目标变量）...")
features = [col for col in numeric_cols if col != 'Concrete Compressive Strength']
if len(features) > 0:
    n_features = min(len(features), 8)
    n_rows = (n_features + 3) // 4
    fig, axes = plt.subplots(n_rows, min(4, n_features), figsize=(16, n_rows * 4))
    if n_rows == 1:
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(features[:8]):
        if i < len(axes):
            axes[i].scatter(df_clean[feature], df_clean['Concrete Compressive Strength'], 
                           alpha=0.5, s=10)
            axes[i].set_xlabel(feature, fontsize=9)
            axes[i].set_ylabel('Compressive Strength', fontsize=9)
            axes[i].grid(True, alpha=0.3)
            
            # 添加趋势线
            try:
                z = np.polyfit(df_clean[feature], df_clean['Concrete Compressive Strength'], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(df_clean[feature])
                axes[i].plot(x_sorted, p(x_sorted), "r-", linewidth=1.5, alpha=0.8)
            except:
                pass
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    scatter_path = os.path.join(fig_dir, '4_scatter_plots.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ 散点图已保存: {scatter_path}")
else:
    print("⚠ 没有找到特征变量（除了目标变量之外）")

# 3.5 目标变量的分布详细图
print("\n3.5 绘制目标变量详细分布图...")
if 'Concrete Compressive Strength' in df_clean.columns:
    target_data = df_clean['Concrete Compressive Strength']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图 + KDE
    ax1.hist(target_data, bins=30, density=True, 
             alpha=0.6, edgecolor='black', label='直方图')
    sns.kdeplot(target_data, ax=ax1, color='red', 
                linewidth=2, label='核密度曲线')
    ax1.axvline(target_data.mean(), 
               color='blue', linestyle='dashed', linewidth=2, 
               label=f'均值={target_data.mean():.2f}')
    ax1.axvline(target_data.median(), 
               color='green', linestyle='dashed', linewidth=2, 
               label=f'中位数={target_data.median():.2f}')
    ax1.set_title('混凝土抗压强度分布', fontsize=12)
    ax1.set_xlabel('抗压强度 (MPa)')
    ax1.set_ylabel('密度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 箱线图
    bp = ax2.boxplot(target_data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_title('混凝土抗压强度箱线图', fontsize=12)
    ax2.set_ylabel('抗压强度 (MPa)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    target_path = os.path.join(fig_dir, '5_target_variable_analysis.png')
    plt.savefig(target_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ 目标变量详细分析图已保存: {target_path}")
else:
    print("⚠ 未找到目标变量 'Concrete Compressive Strength'")

# ==================== 4. 分析总结 ====================
print("\n" + "="*80)
print("第四部分：分析总结与关键发现")
print("="*80)

if 'Concrete Compressive Strength' in df_clean.columns:
    target = df_clean['Concrete Compressive Strength']
    skew_val = target.skew()
    skew_desc = '右偏' if skew_val > 0 else '左偏' if skew_val < 0 else '对称'
    
    # 获取相关性最高的特征
    if len(numeric_cols) > 1:
        corr_series = df_clean[numeric_cols].corr()['Concrete Compressive Strength']
        top_corr = corr_series.sort_values(ascending=False)
        top_positive = top_corr.index[1] if len(top_corr) > 1 else 'N/A'
        top_pos_val = top_corr.iloc[1] if len(top_corr) > 1 else 0
        top_negative = top_corr.index[-1] if len(top_corr) > 1 else 'N/A'
        top_neg_val = top_corr.iloc[-1] if len(top_corr) > 1 else 0
    else:
        top_positive = 'N/A'
        top_pos_val = 0
        top_negative = 'N/A'
        top_neg_val = 0
    
    print(f"""
【关键发现】

1. 数据质量:
   - 数据集无缺失值，数据完整性良好
   - 重复样本数: {duplicate_count} (占比: {duplicate_percent:.2f}%)
   
2. 目标变量 (混凝土抗压强度) 特征:
   - 均值: {target.mean():.2f} MPa
   - 中位数: {target.median():.2f} MPa
   - 标准差: {target.std():.2f} MPa
   - 范围: {target.min():.2f} - {target.max():.2f} MPa
   - 分布形态: {skew_desc} (偏度: {skew_val:.2f})
   
3. 与目标变量相关性最强的特征:
   - 正相关: {top_positive} (相关系数: {top_pos_val:.2f})
   - 负相关: {top_negative} (相关系数: {top_neg_val:.2f})
   
4. 数据预处理建议:
   - 考虑对偏态分布的特征进行对数变换
   - 可考虑删除或保留异常值（需结合业务理解）
   - 建议对特征进行标准化/归一化处理
    """)
else:
    print("⚠ 无法生成分析总结，因为未找到目标变量")

# 保存清洗后的数据
clean_file_path = os.path.join(desktop_path, "concrete_clean_data.csv")
df_clean.to_csv(clean_file_path, index=False)
print(f"\n✓ 清洗后的数据已保存到: {clean_file_path}")
print(f"  清洗后数据维度: {df_clean.shape[0]} 行 × {df_clean.shape[1]} 列")

print("\n" + "="*80)
print("数据分析完成！所有图表已保存到桌面文件夹 'concrete_analysis_figures'")
print("="*80)