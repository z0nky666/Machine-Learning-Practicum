# -*- coding: utf-8 -*-
"""
================================================================================
实验四: MNIST手写数字识别 - CNN卷积神经网络
================================================================================
使用PyTorch CNN对MNIST手写数字(0-9)进行分类

使用模型: DeepSeek-v4-pro
Prompt: "请编写Python代码,使用MNIST手写数字数据集训练一个CNN卷积神经网络。
         网络架构应包含: 2-3个卷积层(Conv2D+BatchNorm+ReLU+MaxPool),
         然后GlobalAvgPool或Flatten,再加2个全连接层(Dropout防过拟合)。
         使用Adam优化器、交叉熵损失、学习率调度器。
         训练10-15个epoch,在测试集上评估准确率、混淆矩阵、分类报告。
         同时绘制训练曲线和混淆矩阵可视化。"
================================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ============================================================
# 路径与参数
# ============================================================
CODE_DIR = Path(r"C:\Users\legion\Desktop\机器学习实训代码\experiment4_mnist")
RESULTS_DIR = CODE_DIR / "results"
FIGURE_DIR = CODE_DIR / "figures"
MODEL_DIR = CODE_DIR / "models"

for d in [RESULTS_DIR, FIGURE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_THREADS = max(1, min(4, os.cpu_count() or 1))
torch.set_num_threads(N_THREADS)

C = ['#9DC8E8','#F3B6C5','#A8D5BA','#C9B6E4','#F5E5A7',
     '#F6C6A8','#B8D0EB','#E8B4B8','#B5D8CC','#D4C5E2']
plt.rcParams.update({'font.sans-serif': ['SimHei','Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 10})


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CNN模型
# ============================================================
class MNIST_CNN(nn.Module):
    """CNN for 28x28 MNIST"""
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


# ============================================================
# 训练函数
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ============================================================
# 可视化
# ============================================================
def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], color='#9DC8E8', linewidth=2, label='Train')
    ax1.plot(history["val_loss"], color='#F3B6C5', linewidth=2, label='Val')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("MNIST CNN - Loss Curves")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], color='#9DC8E8', linewidth=2, label='Train')
    ax2.plot(history["val_acc"], color='#F3B6C5', linewidth=2, label='Val')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("MNIST CNN - Accuracy Curves")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mnist_training_curves.png", dpi=300)
    plt.close()


def plot_confusion_matrix_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')

    for i in range(10):
        for j in range(10):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center', fontsize=8)

    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("MNIST CNN - Normalized Confusion Matrix")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mnist_confusion_matrix.png", dpi=300)
    plt.close()

    return cm


def plot_sample_predictions(model, test_loader):
    """可视化一些预测样本"""
    model.eval()
    X_batch, y_batch = next(iter(test_loader))
    X_batch, y_batch = X_batch[:25].to(DEVICE), y_batch[:25]

    with torch.no_grad():
        preds = model(X_batch).argmax(1).cpu().numpy()

    X_batch = X_batch.cpu()

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_batch[i, 0], cmap='gray')
        color = 'green' if preds[i] == y_batch[i].item() else 'red'
        ax.set_title(f"True:{y_batch[i].item()} Pred:{preds[i]}", color=color, fontsize=9)
        ax.axis('off')
    plt.suptitle("MNIST CNN - Sample Predictions (Green=Correct, Red=Wrong)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mnist_sample_predictions.png", dpi=150)
    plt.close()


# ============================================================
# 主程序
# ============================================================
def main():
    set_seed()
    start = time.time()
    print("="*60)
    print("实验四: MNIST手写数字识别 - CNN")
    print(f"设备: {DEVICE}")
    print("="*60)

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("加载 MNIST 数据集...")
    train_ds = datasets.MNIST(root=str(CODE_DIR / "data"), train=True,
                              download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(CODE_DIR / "data"), train=False,
                             download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    print(f"训练集: {len(train_ds)} 样本, 测试集: {len(test_ds)} 样本")

    # 可视化部分样本
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        img, label = train_ds[i]
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f"Label: {label}", fontsize=8)
        ax.axis('off')
    plt.suptitle("MNIST Sample Images")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mnist_samples.png", dpi=150)
    plt.close()
    print("[OK] 样本图已保存")

    # 构建模型
    model = MNIST_CNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    # 训练
    EPOCHS = 12
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0

    print(f"\n开始训练 ({EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "mnist_cnn_best.pth")

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    train_time = time.time() - start

    # 加载最佳模型评估
    model.load_state_dict(torch.load(MODEL_DIR / "mnist_cnn_best.pth", weights_only=True))
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)

    print(f"\n{'='*60}")
    print(f"最终测试结果:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"训练总时间: {train_time:.1f}s")

    # 详细指标
    report = classification_report(all_labels, all_preds, digits=4,
                                   target_names=[str(i) for i in range(10)])
    print("\n分类报告:")
    print(report)

    # 保存指标
    metrics = {
        "accuracy": float(test_acc),
        "macro_precision": precision_score(all_labels, all_preds, average="macro"),
        "macro_recall": recall_score(all_labels, all_preds, average="macro"),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "train_time": train_time,
        "n_params": n_params,
        "best_val_acc": best_val_acc,
    }
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "mnist_cnn_results.csv",
                                   index=False, encoding="utf-8-sig")

    # 保存分类报告
    with open(RESULTS_DIR / "mnist_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 可视化
    plot_training_curves(history)
    print("[OK] 训练曲线已保存")

    cm = plot_confusion_matrix_cm(all_labels, all_preds)
    print("[OK] 混淆矩阵已保存")

    plot_sample_predictions(model, test_loader)
    print("[OK] 样本预测已保存")

    # Per-class accuracy
    per_class = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = C[:10]
    ax.bar(range(10), per_class, color=colors, edgecolor='white')
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit"); ax.set_ylabel("Accuracy")
    ax.set_title("MNIST CNN - Per-Class Accuracy")
    for i, v in enumerate(per_class):
        ax.text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_ylim(0.95, 1.01)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mnist_per_class_accuracy.png", dpi=300)
    plt.close()

    print(f"\n{'='*60}")
    print("实验四完成!")
    print(f"最佳测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
