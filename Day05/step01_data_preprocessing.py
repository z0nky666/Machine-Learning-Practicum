# -*- coding: utf-8 -*-
"""
================================================================================
实验三 Step 1: UCI HAR 数据加载、EDA与预处理
================================================================================
功能:
  1. 加载561维手工特征 + 原始惯性信号
  2. 数据质量检查与可视化
  3. 特征预处理: 方差过滤、标准化、ANOVA F-score、PCA
  4. 保存处理后的数据供后续模型使用

使用模型: DeepSeek-v4-pro
Prompt: "请编写Python代码加载UCI HAR数据集,进行探索性数据分析(EDA),
         包括标签分布可视化、特征方差分析、ANOVA F-score特征重要性排序、
         Pearson相关性热力图、PCA降维可视化,并将处理后的数据保存为npz格式。
         数据路径为本地UCI HAR Dataset文件夹。"
================================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# ============================================================
# 路径配置
# ============================================================
DATA_ROOT = Path(r"C:\Users\legion\Desktop\机器学习实训\实验三\UCI HAR Dataset")
CODE_DIR = Path(r"C:\Users\legion\Desktop\机器学习实训代码\experiment3_har")
FIGURE_DIR = CODE_DIR / "figures"
PROCESSED_DIR = CODE_DIR / "processed_data"
RESULTS_DIR = CODE_DIR / "results"

for d in [FIGURE_DIR, PROCESSED_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TOP_K = 100
PCA_VARIANCE = 0.95

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def make_feature_names_unique(names):
    seen = {}
    result = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            result.append(n)
        else:
            seen[n] += 1
            result.append(f"{n}__dup{seen[n]}")
    return result


def load_har_data():
    """加载UCI HAR 561维特征数据"""
    print("=" * 70)
    print("1. 加载UCI HAR Dataset")
    print("=" * 70)

    features_df = pd.read_csv(DATA_ROOT / "features.txt", sep=r"\s+",
                              header=None, names=["id", "name"], engine="python")
    raw_names = features_df["name"].tolist()
    unique_names = make_feature_names_unique(raw_names)

    activity_df = pd.read_csv(DATA_ROOT / "activity_labels.txt", sep=r"\s+",
                              header=None, names=["id", "name"], engine="python")
    activity_map = dict(zip(activity_df["id"], activity_df["name"]))

    # 训练集
    X_train = pd.read_csv(DATA_ROOT / "train" / "X_train.txt", sep=r"\s+",
                          header=None, dtype=np.float32, engine="python")
    X_train.columns = unique_names
    y_train = pd.read_csv(DATA_ROOT / "train" / "y_train.txt",
                          header=None).iloc[:, 0].astype(int).to_numpy()
    subj_train = pd.read_csv(DATA_ROOT / "train" / "subject_train.txt",
                             header=None).iloc[:, 0].astype(int).to_numpy()

    # 测试集
    X_test = pd.read_csv(DATA_ROOT / "test" / "X_test.txt", sep=r"\s+",
                         header=None, dtype=np.float32, engine="python")
    X_test.columns = unique_names
    y_test = pd.read_csv(DATA_ROOT / "test" / "y_test.txt",
                         header=None).iloc[:, 0].astype(int).to_numpy()
    subj_test = pd.read_csv(DATA_ROOT / "test" / "subject_test.txt",
                            header=None).iloc[:, 0].astype(int).to_numpy()

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"标签范围: {y_train.min()}-{y_train.max()}, 受试者: {len(np.unique(subj_train))}")

    return X_train, X_test, y_train, y_test, subj_train, subj_test, unique_names, activity_map


def load_raw_signals():
    """加载原始惯性信号 (用于1D-CNN/LSTM/GRU)"""
    print("\n" + "=" * 70)
    print("2. 加载原始惯性信号 (128时间步 × 9通道)")
    print("=" * 70)

    signals = ["body_acc_x", "body_acc_y", "body_acc_z",
               "body_gyro_x", "body_gyro_y", "body_gyro_z",
               "total_acc_x", "total_acc_y", "total_acc_z"]

    X_train_raw, X_test_raw = [], []

    for sig in signals:
        train_file = DATA_ROOT / "train" / "Inertial Signals" / f"{sig}_train.txt"
        test_file = DATA_ROOT / "test" / "Inertial Signals" / f"{sig}_test.txt"

        train_data = pd.read_csv(train_file, sep=r"\s+", header=None,
                                 dtype=np.float32, engine="python").to_numpy()
        test_data = pd.read_csv(test_file, sep=r"\s+", header=None,
                                dtype=np.float32, engine="python").to_numpy()
        X_train_raw.append(train_data)
        X_test_raw.append(test_data)

    # (samples, 128, 9)
    X_train_raw = np.stack(X_train_raw, axis=-1)
    X_test_raw = np.stack(X_test_raw, axis=-1)

    print(f"原始信号训练集: {X_train_raw.shape}, 测试集: {X_test_raw.shape}")
    return X_train_raw, X_test_raw


def eda_visualization(X_train, y_train, activity_map, feature_names):
    """EDA: 标签分布、特征方差、ANOVA Top-20、相关性热力图"""
    print("\n" + "=" * 70)
    print("3. 探索性数据分析与可视化")
    print("=" * 70)

    # --- 标签分布 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    unique, counts = np.unique(y_train, return_counts=True)
    names = [activity_map[u] for u in unique]
    colors = ['#9DC8E8', '#F3B6C5', '#A8D5BA', '#C9B6E4', '#F5E5A7', '#F6C6A8']
    axes[0].bar(names, counts, color=colors, edgecolor='white')
    axes[0].set_title("Training Set Activity Distribution")
    axes[0].set_ylabel("Samples")
    axes[0].tick_params(axis='x', rotation=25)
    for i, v in enumerate(counts):
        axes[0].text(i, v + 10, str(v), ha='center', fontsize=9)

    axes[1].pie(counts, labels=names, colors=colors, autopct='%1.1f%%')
    axes[1].set_title("Activity Proportion")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_activity_distribution.png", dpi=300)
    plt.close()
    print("✓ 活动分布图已保存")

    # --- ANOVA F-score Top-20 ---
    X_arr = X_train.to_numpy(dtype=np.float32)
    f_scores, p_values = f_classif(X_arr, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    top_idx = np.argsort(f_scores)[-20:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_scores = f_scores[top_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(20), top_scores[::-1], color='#9DC8E8', edgecolor='white')
    ax.set_yticks(range(20))
    ax.set_yticklabels([top_names[i] for i in range(20)[::-1]], fontsize=7)
    ax.set_xlabel("ANOVA F-score")
    ax.set_title("Top 20 Features by ANOVA F-score")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_top20_anova.png", dpi=300)
    plt.close()
    print("✓ ANOVA Top-20特征图已保存")

    # --- 相关性热力图 (Top-30) ---
    top30_idx = np.argsort(f_scores)[-30:]
    top30_names = [feature_names[i] for i in top30_idx]
    top30_data = X_arr[:, top30_idx]
    corr = np.corrcoef(top30_data.T)

    fig, ax = plt.subplots(figsize=(15, 12))
    im = ax.imshow(corr, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(30))
    ax.set_yticks(range(30))
    ax.set_xticklabels(top30_names, rotation=90, fontsize=6)
    ax.set_yticklabels(top30_names, fontsize=6)
    ax.set_title("Pearson Correlation of Top 30 ANOVA Features")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_correlation_heatmap.png", dpi=300)
    plt.close()
    print("✓ 相关性热力图已保存")


def preprocess_features(X_train_df, X_test_df, y_train, feature_names):
    """特征预处理: 方差过滤 -> 标准化 -> Top-K选择 -> PCA"""
    print("\n" + "=" * 70)
    print("4. 特征预处理")
    print("=" * 70)

    X_train_raw = X_train_df.to_numpy(dtype=np.float32)
    X_test_raw = X_test_df.to_numpy(dtype=np.float32)

    # 4.1 方差过滤
    var_sel = VarianceThreshold(threshold=0.0)
    X_train_vf = var_sel.fit_transform(X_train_raw)
    X_test_vf = var_sel.transform(X_test_raw)
    kept_mask = var_sel.get_support()
    kept_names = np.array(feature_names)[kept_mask]
    n_removed = X_train_raw.shape[1] - X_train_vf.shape[1]
    print(f"方差过滤: {X_train_raw.shape[1]} -> {X_train_vf.shape[1]} 特征 (移除{n_removed}个零方差特征)")

    # 4.2 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_vf)
    X_test_scaled = scaler.transform(X_test_vf)
    print(f"标准化完成: 均值={X_train_scaled.mean():.6f}, 标准差={X_train_scaled.std():.6f}")

    # 4.3 ANOVA Top-K
    actual_k = min(TOP_K, X_train_scaled.shape[1])
    selector = SelectKBest(score_func=f_classif, k=actual_k)
    X_train_topk = selector.fit_transform(X_train_scaled, y_train)
    X_test_topk = selector.transform(X_test_scaled)
    topk_mask = selector.get_support()
    topk_names = kept_names[topk_mask]
    print(f"Top-{actual_k}特征选择完成")

    # 4.4 PCA
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)[-1]
    print(f"PCA: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]}维 (累计解释方差={cum_var:.4f})")

    # PCA可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 累计方差
    axes[0].plot(np.cumsum(pca.explained_variance_ratio_), color='#9DC8E8', linewidth=2)
    axes[0].axhline(y=PCA_VARIANCE, color='#F3B6C5', linestyle='--', label=f'{PCA_VARIANCE:.0%}')
    axes[0].set_xlabel("Components"); axes[0].set_ylabel("Cumulative Variance")
    axes[0].set_title("PCA Cumulative Explained Variance")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 2D散点
    pca2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2d.fit_transform(X_train_scaled)
    colors = ['#9DC8E8', '#F3B6C5', '#A8D5BA', '#C9B6E4', '#F5E5A7', '#F6C6A8']
    for i, act_id in enumerate(sorted(np.unique(y_train))):
        mask = y_train == act_id
        axes[1].scatter(X_2d[mask, 0][:500], X_2d[mask, 1][:500],
                       c=colors[i], label=f'Class {act_id}', alpha=0.6, s=10)
    axes[1].set_title("PCA 2D Visualization (sampled)")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_pca_visualization.png", dpi=300)
    plt.close()
    print("✓ PCA可视化已保存")

    # 保存
    y_train_0b = y_train - 1
    y_test_0b = X_test_df.shape[0]  # placeholder

    # 完整特征
    np.savez_compressed(PROCESSED_DIR / "full_features.npz",
        X_train_raw=X_train_vf, X_test_raw=X_test_vf,
        X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
        y_train=y_train, y_test=np.zeros(1),  # will be updated
        feature_names=kept_names)
    print("✓ full_features.npz 已保存")

    return (X_train_vf, X_test_vf, X_train_scaled, X_test_scaled,
            X_train_topk, X_test_topk, X_train_pca, X_test_pca,
            kept_names, topk_names, scaler, var_sel, selector, pca)


def main():
    start = time.time()
    print("=" * 70)
    print("实验三 Step 1: UCI HAR 数据预处理")
    print(f"数据路径: {DATA_ROOT}")
    print(f"输出路径: {CODE_DIR}")
    print("=" * 70)

    # 加载数据
    X_train_df, X_test_df, y_train, y_test, subj_train, subj_test, feat_names, act_map = load_har_data()

    # 加载原始信号
    X_train_signal, X_test_signal = load_raw_signals()

    # 保存原始信号
    np.savez_compressed(PROCESSED_DIR / "raw_signals.npz",
        X_train=X_train_signal, X_test=X_test_signal,
        y_train=y_train, y_test=y_test,
        subject_train=subj_train, subject_test=subj_test)
    print("✓ raw_signals.npz 已保存")

    # 保存标签和受试者信息
    np.savez_compressed(PROCESSED_DIR / "labels_subjects.npz",
        y_train=y_train, y_test=y_test,
        subject_train=subj_train, subject_test=subj_test,
        y_train_zero=y_train - 1, y_test_zero=y_test - 1)

    # EDA
    eda_visualization(X_train_df, y_train, act_map, feat_names)

    # 预处理
    result = preprocess_features(X_train_df, X_test_df, y_train, feat_names)

    # 保存活动映射
    act_df = pd.DataFrame([(k, v) for k, v in act_map.items()],
                          columns=["id", "activity"])
    act_df.to_csv(RESULTS_DIR / "activity_labels.csv", index=False, encoding="utf-8-sig")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"Step 1 完成! 总耗时: {elapsed:.1f}s")
    print(f"所有输出保存在: {CODE_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
