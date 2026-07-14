# -*- coding: utf-8 -*-
"""
================================================================================
实验三 Step 4: 模型对比分析
================================================================================
汇总所有7个模型的结果,生成综合对比图表

使用模型: DeepSeek-v4-pro
Prompt: "请编写Python代码,汇总UCI HAR数据集上7个模型(GBDT、XGBoost、
         LightGBM、MLP、1D-CNN、LSTM、GRU)的测试结果,生成综合对比:
         1. 准确率/Macro-F1/训练时间的横向柱状图对比
         2. 模型性能雷达图
         3. 训练时间vs准确率散点图
         4. 汇总表格
         所有图表使用论文风格配色。"
================================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(r"C:\Users\legion\Desktop\机器学习实训代码\experiment3_har")
RESULTS_DIR = CODE_DIR / "results"
FIGURE_DIR = CODE_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

C = ['#9DC8E8','#F3B6C5','#A8D5BA','#C9B6E4','#F5E5A7','#F6C6A8','#B8D0EB']
TEXT = '#3F4A54'

plt.rcParams.update({'font.sans-serif': ['SimHei','Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 10,
                     'axes.titlesize': 13, 'axes.labelsize': 11})


def apply_style(ax):
    ax.grid(True, linestyle='--', linewidth=0.7, color='#E7E9ED', alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def load_results():
    """加载所有模型结果"""
    rows = []

    trad = pd.read_csv(RESULTS_DIR / "traditional_ml_results.csv")
    for _, r in trad.iterrows():
        rows.append({"model": r["model"], "accuracy": r["accuracy"],
                     "macro_f1": r["macro_f1"], "train_time": r["train_time"]})

    dl = pd.read_csv(RESULTS_DIR / "deep_learning_results.csv")
    for _, r in dl.iterrows():
        rows.append({"model": r["model"], "accuracy": r["accuracy"],
                     "macro_f1": r["macro_f1"], "train_time": r["train_time"]})

    return pd.DataFrame(rows)


def main():
    print("="*60)
    print("实验三 Step 4: 模型对比分析")
    print("="*60)

    df = load_results()
    print("\n汇总结果:")
    print(df.to_string(index=False))

    # ---- 1. 准确率 & Macro-F1 对比柱状图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    metrics = ["accuracy", "macro_f1"]
    titles = ["Test Accuracy", "Test Macro-F1"]
    for ax, metric, title in zip([ax1, ax2], metrics, titles):
        sorted_df = df.sort_values(metric)
        bars = ax.barh(sorted_df["model"], sorted_df[metric],
                       color=C[:len(df)], edgecolor='white', height=0.6)
        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9, color=TEXT)
        ax.set_title(title)
        ax.set_xlim(min(sorted_df[metric]) - 0.02, max(sorted_df[metric]) + 0.02)
        apply_style(ax)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "comparison_accuracy_f1.png", dpi=300)
    plt.close()
    print("[OK] accuracy_f1 comparison saved")

    # ---- 2. 训练时间对比 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_df = df.sort_values("train_time")
    bars = ax.barh(sorted_df["model"], sorted_df["train_time"],
                   color=C[:len(df)], edgecolor='white', height=0.6)
    for bar, val in zip(bars, sorted_df["train_time"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}s', va='center', fontsize=9, color=TEXT)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Model Training Time Comparison")
    apply_style(ax)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "comparison_training_time.png", dpi=300)
    plt.close()
    print("[OK] training time comparison saved")

    # ---- 3. 综合对比散点图 ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row["train_time"], row["accuracy"], s=200,
                  color=C[i % len(C)], edgecolor='white', linewidth=1.5, zorder=5)
        offset = (10, -0.003) if row["model"] != "LSTM" else (10, 0.002)
        ax.annotate(row["model"], (row["train_time"], row["accuracy"]),
                    textcoords="offset points", xytext=offset, fontsize=9, color=TEXT)
    ax.set_xlabel("Training Time (s)"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Performance: Accuracy vs Training Time")
    apply_style(ax)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "comparison_accuracy_vs_time.png", dpi=300)
    plt.close()
    print("[OK] accuracy vs time scatter saved")

    # ---- 4. 雷达图 ----
    radar_metrics = ["accuracy", "macro_f1", "weighted_f1"]
    available_metrics = [m for m in radar_metrics if m in df.columns]
    if len(available_metrics) >= 2:
        df_radar = df.set_index("model")[available_metrics]
        df_norm = (df_radar - 0.85) / 0.15

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]

        for i, (model, row) in enumerate(df_norm.iterrows()):
            values = row.values.tolist() + [row.values[0]]
            ax.fill(angles, values, alpha=0.15, color=C[i % len(C)])
            ax.plot(angles, values, 'o-', linewidth=2, color=C[i % len(C)], label=model)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics)
        ax.set_yticklabels([])
        ax.set_title("Model Performance Radar (normalized)")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "comparison_radar.png", dpi=300)
        plt.close()
        print("[OK] radar chart saved")

    # ---- 5. 汇总表格保存 ----
    summary = df.copy()
    summary = summary.sort_values("accuracy", ascending=False)
    summary.to_csv(RESULTS_DIR / "all_models_comparison.csv", index=False, encoding="utf-8-sig")

    # 找出最优模型
    best_acc = df.loc[df["accuracy"].idxmax()]
    best_f1 = df.loc[df["macro_f1"].idxmax()]
    fastest = df.loc[df["train_time"].idxmin()]

    print(f"\n=== 综合推荐 ===")
    print(f"最高准确率: {best_acc['model']} ({best_acc['accuracy']:.4f})")
    print(f"最高Macro-F1: {best_f1['model']} ({best_f1['macro_f1']:.4f})")
    print(f"最快训练: {fastest['model']} ({fastest['train_time']:.1f}s)")

    # 综合评分 (准确率 + F1 均权, 同时考虑训练时间)
    df["score"] = (df["accuracy"] + df["macro_f1"]) / 2
    best = df.loc[df["score"].idxmax()]
    print(f"综合最优: {best['model']} (综合分={best['score']:.4f})")

    print(f"\nStep 4 完成!")


if __name__ == "__main__":
    main()
