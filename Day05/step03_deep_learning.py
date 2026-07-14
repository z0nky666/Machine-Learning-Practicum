# -*- coding: utf-8 -*-
"""
================================================================================
实验三 Step 3: 深度学习模型训练 (MLP, 1D-CNN, LSTM, GRU)
================================================================================
MLP使用sklearn在561维特征上训练
1D-CNN/LSTM/GRU使用PyTorch在原始128x9时序信号上训练

使用模型: DeepSeek-v4-pro
Prompt: "请编写Python代码,使用UCI HAR数据集分别训练MLP、1D-CNN、LSTM、GRU模型。
         MLP使用561维手工特征,1D-CNN/LSTM/GRU使用128时间步x9通道的原始惯性信号。
         请使用PyTorch实现深度学习模型,包含合理的网络架构:
         - MLP: 多层全连接+Dropout+BatchNorm
         - 1D-CNN: 3层Conv1D+MaxPool+GlobalAvgPool+FC
         - LSTM: 2层双向LSTM+FC
         - GRU: 2层GRU+FC
         训练时使用Adam优化器、交叉熵损失、学习率衰减。
         评估在官方测试集上的准确率、Macro-F1,并绘制混淆矩阵和训练曲线。"
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

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 路径与参数
# ============================================================
DATA_ROOT = Path(r"C:\Users\legion\Desktop\机器学习实训\实验三\UCI HAR Dataset")
CODE_DIR = Path(r"C:\Users\legion\Desktop\机器学习实训代码\experiment3_har")
PROCESSED_DIR = CODE_DIR / "processed_data"
RESULTS_DIR = CODE_DIR / "results"
FIGURE_DIR = CODE_DIR / "figures"
MODEL_DIR = CODE_DIR / "models"

for d in [RESULTS_DIR, FIGURE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_THREADS = max(1, min(4, os.cpu_count() or 1))
torch.set_num_threads(N_THREADS)

C = ['#9DC8E8','#F3B6C5','#A8D5BA','#C9B6E4','#F5E5A7','#F6C6A8']
plt.rcParams.update({'font.sans-serif': ['SimHei','Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus': False})


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def load_activity_map():
    df = pd.read_csv(DATA_ROOT / "activity_labels.txt", sep=r"\s+",
                     header=None, names=["id","name"], engine="python")
    return dict(zip(df["id"], df["name"]))


def plot_confusion(y_true, y_pred, act_map, title, fname):
    names = [act_map[i] for i in sorted(act_map.keys())]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(names, rotation=30, ha='right'); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(FIGURE_DIR / fname, dpi=300); plt.close()


def plot_training_curves(train_loss, train_acc, title, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_loss, color=C[0], linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title(f"{title} - Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_acc, color=C[1], linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title(f"{title} - Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / fname, dpi=300); plt.close()


# ============================================================
# MLP (sklearn)
# ============================================================
def train_mlp(X_train, X_test, y_train, y_test, act_map):
    print("\n" + "="*70)
    print("MLP (多层感知机) 训练")
    print("="*70)

    # 使用标准化后的特征
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu', solver='adam',
        alpha=0.0001, batch_size=128,
        learning_rate='adaptive', learning_rate_init=0.001,
        max_iter=100, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=10,
        random_state=RANDOM_STATE, verbose=False)

    t0 = time.time()
    model.fit(X_train_s, y_train - 1)  # 0-based labels
    train_time = time.time() - t0

    y_pred = model.predict(X_test_s)
    y_test_0 = y_test - 1
    metrics = calc_metrics(y_test_0, y_pred)
    metrics["train_time"] = train_time
    metrics["model"] = "MLP"

    plot_confusion(y_test_0, y_pred, {i: act_map[i+1] for i in range(6)},
                   "MLP - Normalized Confusion Matrix", "mlp_confusion_matrix.png")

    # Loss curve
    if hasattr(model, 'loss_curve_'):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(model.loss_curve_, color=C[0], linewidth=2)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
        ax.set_title("MLP Training Loss Curve")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "mlp_training_curve.png", dpi=300); plt.close()

    joblib.dump(model, MODEL_DIR / "mlp_model.joblib")
    print(f"MLP - Test Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, Time={train_time:.1f}s")
    return metrics


# ============================================================
# PyTorch 深度学习模型定义
# ============================================================
class CNN1D(nn.Module):
    def __init__(self, n_channels=9, n_classes=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 128, 9) -> (B, 9, 128)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class LSTMNet(nn.Module):
    def __init__(self, n_features=9, n_classes=6, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden * 2, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUNet(nn.Module):
    def __init__(self, n_features=9, n_classes=6, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2,
                         batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden * 2, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ============================================================
# 训练函数
# ============================================================
def train_torch_model(model, train_loader, X_test_t, y_test_t, act_map,
                      model_name, epochs=50, lr=0.001):
    """训练PyTorch模型"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    train_losses, train_accs = [], []

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_b.size(0)
            correct += (out.argmax(1) == y_b).sum().item()
            total += X_b.size(0)

        avg_loss = total_loss / total
        avg_acc = correct / total
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    train_time = time.time() - t0

    # 测试
    model.eval()
    with torch.no_grad():
        X_test_t = X_test_t.to(DEVICE)
        out = model(X_test_t)
        y_pred = out.argmax(1).cpu().numpy()

    y_test_np = y_test_t.numpy() if hasattr(y_test_t, 'numpy') else y_test_t
    metrics = calc_metrics(y_test_np, y_pred)
    metrics["train_time"] = train_time
    metrics["model"] = model_name

    # 绘图
    plot_training_curves(train_losses, train_accs, model_name,
                         f"{model_name.lower()}_training_curve.png")
    act_map_0 = {i: act_map[i+1] for i in range(6)}
    plot_confusion(y_test_np, y_pred, act_map_0,
                   f"{model_name} - Normalized Confusion Matrix",
                   f"{model_name.lower()}_confusion_matrix.png")

    # 保存模型
    torch.save(model.state_dict(), MODEL_DIR / f"{model_name.lower()}_model.pth")

    print(f"{model_name} - Test Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, Time={train_time:.1f}s")
    return metrics


# ============================================================
# 主程序
# ============================================================
def main():
    set_seed()
    start = time.time()
    print("="*70)
    print("实验三 Step 3: 深度学习模型 (MLP, 1D-CNN, LSTM, GRU)")
    print(f"设备: {DEVICE}, CPU线程: {N_THREADS}")
    print("="*70)

    act_map = load_activity_map()

    # ---- 加载特征数据 (用于MLP) ----
    data_f = np.load(PROCESSED_DIR / "full_features.npz", allow_pickle=False)
    label_data = np.load(PROCESSED_DIR / "labels_subjects.npz", allow_pickle=False)
    X_train_f = data_f["X_train_raw"].astype(np.float32)
    X_test_f = data_f["X_test_raw"].astype(np.float32)
    y_train, y_test = label_data["y_train"].astype(np.int64), label_data["y_test"].astype(np.int64)

    # ---- 加载原始信号 (用于CNN/LSTM/GRU) ----
    data_s = np.load(PROCESSED_DIR / "raw_signals.npz", allow_pickle=False)
    X_train_s = data_s["X_train"].astype(np.float32)
    X_test_s = data_s["X_test"].astype(np.float32)

    # 标准化原始信号
    mean_s = X_train_s.mean(axis=(0,1), keepdims=True)
    std_s = X_train_s.std(axis=(0,1), keepdims=True) + 1e-8
    X_train_s = (X_train_s - mean_s) / std_s
    X_test_s = (X_test_s - mean_s) / std_s

    print(f"特征数据: train={X_train_f.shape}, test={X_test_f.shape}")
    print(f"信号数据: train={X_train_s.shape}, test={X_test_s.shape}")

    all_results = []

    # ---- MLP ----
    set_seed()
    mlp_m = train_mlp(X_train_f, X_test_f, y_train, y_test, act_map)
    all_results.append(mlp_m)

    # ---- 1D-CNN ----
    print("\n" + "="*70)
    print("1D-CNN 训练")
    print("="*70)
    set_seed()
    y_train_0 = y_train - 1; y_test_0 = y_test - 1
    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train_0))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    cnn = CNN1D()
    cnn_m = train_torch_model(cnn, train_loader,
                              torch.FloatTensor(X_test_s),
                              y_test_0, act_map, "1D-CNN", epochs=40)
    all_results.append(cnn_m)

    # ---- LSTM ----
    print("\n" + "="*70)
    print("LSTM 训练")
    print("="*70)
    set_seed()
    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train_0))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    lstm = LSTMNet()
    lstm_m = train_torch_model(lstm, train_loader,
                               torch.FloatTensor(X_test_s),
                               y_test_0, act_map, "LSTM", epochs=40)
    all_results.append(lstm_m)

    # ---- GRU ----
    print("\n" + "="*70)
    print("GRU 训练")
    print("="*70)
    set_seed()
    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train_0))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    gru = GRUNet()
    gru_m = train_torch_model(gru, train_loader,
                              torch.FloatTensor(X_test_s),
                              y_test_0, act_map, "GRU", epochs=40)
    all_results.append(gru_m)

    # 保存
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "deep_learning_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n深度学习结果已保存:")
    print(df[["model","accuracy","macro_f1","weighted_f1","train_time"]].to_string(index=False))

    elapsed = time.time() - start
    print(f"\nStep 3 完成! 总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
