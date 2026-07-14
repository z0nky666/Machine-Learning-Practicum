# -*- coding: utf-8 -*-
"""
================================================================================
实验三 Step 2: 传统机器学习模型训练 (GBDT, XGBoost, LightGBM)
================================================================================
使用561维手工特征,基于受试者划分进行验证,选择最优模型并在官方测试集评估

使用模型: DeepSeek-v4-pro
Prompt: "请编写Python代码,使用UCI HAR数据集的561维特征,
         分别训练GBDT、XGBoost、LightGBM三个集成学习模型。
         使用受试者感知的验证集划分策略(GroupShuffleSplit),
         每个模型设计3组候选超参数,基于验证集Macro-F1选择最优,
         最后在官方测试集上评估。需要输出混淆矩阵、分类报告、
         特征重要性排名和训练曲线。请使用tqdm显示训练进度。"
================================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import lightgbm as lgb
import joblib

# ============================================================
# 路径配置
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
N_JOBS = max(1, min(4, os.cpu_count() or 1))
VALIDATION_RATIO = 0.20

# 马卡龙配色
C = ['#9DC8E8','#F3B6C5','#A8D5BA','#C9B6E4','#F5E5A7','#F6C6A8']
TEXT_COLOR = '#3F4A54'
GRID_COLOR = '#E7E9ED'

plt.rcParams.update({'font.sans-serif': ['SimHei','Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus': False, 'figure.facecolor': 'white'})


def apply_style(ax):
    ax.grid(True, linestyle='--', linewidth=0.7, color=GRID_COLOR, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def calc_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None:
        try:
            m["macro_roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except:
            m["macro_roc_auc"] = np.nan
    return m


def load_data():
    data = np.load(PROCESSED_DIR / "full_features.npz", allow_pickle=False)
    label_data = np.load(PROCESSED_DIR / "labels_subjects.npz", allow_pickle=False)
    return (data["X_train_raw"].astype(np.float32),
            data["X_test_raw"].astype(np.float32),
            label_data["y_train"].astype(np.int64),
            label_data["y_test"].astype(np.int64),
            label_data["subject_train"].astype(np.int64),
            label_data["subject_test"].astype(np.int64),
            data["feature_names"].astype(str),
            label_data["y_train_zero"].astype(np.int64))


def load_activity_map():
    df = pd.read_csv(DATA_ROOT / "activity_labels.txt", sep=r"\s+",
                     header=None, names=["id","name"], engine="python")
    return dict(zip(df["id"], df["name"]))


def get_split(X_train, y_train, subj_train):
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_RATIO, random_state=RANDOM_STATE)
    fit_idx, val_idx = next(splitter.split(X_train, y_train, groups=subj_train))
    return fit_idx, val_idx


# ============================================================
# GBDT
# ============================================================
def train_gbdt(X_train, y_train, subj_train, X_test, y_test, act_map, feat_names):
    print("\n" + "="*70)
    print("GBDT 模型训练")
    print("="*70)

    configs = [
        {"name":"GBDT_Light","n_estimators":30,"lr":0.12,"max_depth":2,"min_samples_leaf":4,"subsample":0.80},
        {"name":"GBDT_Balanced","n_estimators":45,"lr":0.10,"max_depth":2,"min_samples_leaf":2,"subsample":0.85},
        {"name":"GBDT_Deep","n_estimators":60,"lr":0.08,"max_depth":3,"min_samples_leaf":2,"subsample":0.85},
    ]

    fit_idx, val_idx = get_split(X_train, y_train, subj_train)
    X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    best_model, best_cfg, best_f1 = None, None, -1
    results = []

    for cfg in configs:
        print(f"\n训练 {cfg['name']}...")
        model = GradientBoostingClassifier(
            n_estimators=cfg["n_estimators"], learning_rate=cfg["lr"],
            max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
            subsample=cfg["subsample"], max_features="sqrt",
            validation_fraction=0.1, n_iter_no_change=6, tol=1e-3,
            random_state=RANDOM_STATE)
        t0 = time.time()
        model.fit(X_fit, y_fit)
        t = time.time() - t0

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)
        m = calc_metrics(y_val, y_pred, y_prob)
        m["train_time"] = t
        m["model"] = cfg["name"]
        results.append(m)
        print(f"  Val Accuracy={m['accuracy']:.4f}, Macro-F1={m['macro_f1']:.4f}, Time={t:.1f}s")

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]; best_model = model; best_cfg = cfg

    # 最终训练
    final_cfg = best_cfg.copy()
    actual_n = min(getattr(best_model, 'n_estimators_', best_cfg["n_estimators"]), best_cfg["n_estimators"])
    final_cfg["n_estimators"] = max(1, actual_n)

    final_model = GradientBoostingClassifier(
        n_estimators=final_cfg["n_estimators"], learning_rate=final_cfg["lr"],
        max_depth=final_cfg["max_depth"], min_samples_leaf=final_cfg["min_samples_leaf"],
        subsample=final_cfg["subsample"], max_features="sqrt", random_state=RANDOM_STATE)

    t0 = time.time()
    final_model.fit(X_train, y_train)
    final_time = time.time() - t0

    # 测试集评估
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)
    test_m = calc_metrics(y_test, y_pred, y_prob)
    test_m["train_time"] = final_time
    test_m["model"] = "GBDT"

    # 混淆矩阵
    names = [act_map[i] for i in sorted(act_map.keys())]
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(9,7))
    im = ax.imshow(cm/cm.sum(axis=1,keepdims=True), cmap='Blues', aspect='auto')
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(names, rotation=30, ha='right'); ax.set_yticklabels(names)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{cm[i,j]/cm.sum(axis=1)[i]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title("GBDT - Normalized Confusion Matrix")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "gbdt_confusion_matrix.png", dpi=300); plt.close()

    # 特征重要性
    imp = pd.DataFrame({"feature": feat_names, "importance": final_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.barh(range(20), imp["importance"].values[::-1], color=C[0], edgecolor='white')
    ax.set_yticks(range(20)); ax.set_yticklabels(imp["feature"].values[::-1], fontsize=7)
    ax.set_xlabel("Importance"); ax.set_title("GBDT Top-20 Feature Importance")
    apply_style(ax); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "gbdt_feature_importance.png", dpi=300); plt.close()

    joblib.dump(final_model, MODEL_DIR / "gbdt_model.joblib")
    print(f"GBDT final - Test Accuracy={test_m['accuracy']:.4f}, Macro-F1={test_m['macro_f1']:.4f}, Time={final_time:.1f}s")

    return test_m, results


# ============================================================
# XGBoost
# ============================================================
def train_xgboost(X_train, y_train, subj_train, X_test, y_test, act_map, feat_names):
    print("\n" + "="*70)
    print("XGBoost 模型训练")
    print("="*70)

    y_train_0 = y_train - 1
    y_test_0 = y_test - 1

    configs = [
        {"name":"XGB_Light","n_estimators":50,"lr":0.12,"max_depth":3,"min_child_weight":2,"subsample":0.80,"colsample":0.70,"gamma":0.0,"reg_lambda":1.5,"reg_alpha":0.0},
        {"name":"XGB_Balanced","n_estimators":75,"lr":0.10,"max_depth":3,"min_child_weight":1,"subsample":0.85,"colsample":0.80,"gamma":0.0,"reg_lambda":1.0,"reg_alpha":0.0},
        {"name":"XGB_Deep","n_estimators":100,"lr":0.08,"max_depth":4,"min_child_weight":1,"subsample":0.85,"colsample":0.80,"gamma":0.05,"reg_lambda":1.5,"reg_alpha":0.0},
    ]

    fit_idx, val_idx = get_split(X_train, y_train_0, subj_train)
    X_fit, y_fit = X_train[fit_idx], y_train_0[fit_idx]
    X_val, y_val = X_train[val_idx], y_train_0[val_idx]

    best_model, best_cfg, best_f1 = None, None, -1
    results = []

    for cfg in configs:
        print(f"\n训练 {cfg['name']}...")
        model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=NUM_CLASSES,
            n_estimators=cfg["n_estimators"], learning_rate=cfg["lr"],
            max_depth=cfg["max_depth"], min_child_weight=cfg["min_child_weight"],
            subsample=cfg["subsample"], colsample_bytree=cfg["colsample"],
            gamma=cfg["gamma"], reg_lambda=cfg["reg_lambda"], reg_alpha=cfg["reg_alpha"],
            tree_method="hist", eval_metric="mlogloss", n_jobs=N_JOBS,
            random_state=RANDOM_STATE, verbosity=0,
            early_stopping_rounds=10)

        t0 = time.time()
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        t = time.time() - t0

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)
        m = calc_metrics(y_val, y_pred, y_prob)
        m["train_time"] = t; m["model"] = cfg["name"]
        results.append(m)
        print(f"  Val Accuracy={m['accuracy']:.4f}, Macro-F1={m['macro_f1']:.4f}, Time={t:.1f}s")

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]; best_model = model; best_cfg = cfg

    # 最终训练
    n_rounds = getattr(best_model, 'best_iteration', best_cfg["n_estimators"])
    if n_rounds is None: n_rounds = best_cfg["n_estimators"]
    n_rounds = max(1, min(n_rounds + 1, best_cfg["n_estimators"]))

    final_model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=NUM_CLASSES,
        n_estimators=n_rounds, learning_rate=best_cfg["lr"],
        max_depth=best_cfg["max_depth"], min_child_weight=best_cfg["min_child_weight"],
        subsample=best_cfg["subsample"], colsample_bytree=best_cfg["colsample"],
        gamma=best_cfg["gamma"], reg_lambda=best_cfg["reg_lambda"], reg_alpha=best_cfg["reg_alpha"],
        tree_method="hist", n_jobs=N_JOBS, random_state=RANDOM_STATE, verbosity=0)

    t0 = time.time()
    final_model.fit(X_train, y_train_0, verbose=False)
    final_time = time.time() - t0

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)
    test_m = calc_metrics(y_test_0, y_pred, y_prob)
    test_m["train_time"] = final_time; test_m["model"] = "XGBoost"

    # 混淆矩阵
    act_map_0 = {i: act_map[i+1] for i in range(6)}
    names = [act_map_0[i] for i in range(6)]
    cm = confusion_matrix(y_test_0, y_pred)
    fig, ax = plt.subplots(figsize=(9,7))
    im = ax.imshow(cm/cm.sum(axis=1,keepdims=True), cmap='Blues', aspect='auto')
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(names, rotation=30, ha='right'); ax.set_yticklabels(names)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{cm[i,j]/cm.sum(axis=1)[i]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title("XGBoost - Normalized Confusion Matrix")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_confusion_matrix.png", dpi=300); plt.close()

    # 特征重要性
    imp = pd.DataFrame({"feature": feat_names, "importance": final_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.barh(range(20), imp["importance"].values[::-1], color=C[1], edgecolor='white')
    ax.set_yticks(range(20)); ax.set_yticklabels(imp["feature"].values[::-1], fontsize=7)
    ax.set_xlabel("Importance"); ax.set_title("XGBoost Top-20 Feature Importance")
    apply_style(ax); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_feature_importance.png", dpi=300); plt.close()

    final_model.save_model(str(MODEL_DIR / "xgboost_model.json"))
    print(f"XGBoost final - Test Accuracy={test_m['accuracy']:.4f}, Macro-F1={test_m['macro_f1']:.4f}, Time={final_time:.1f}s")

    return test_m, results


# ============================================================
# LightGBM
# ============================================================
def train_lightgbm(X_train, y_train, subj_train, X_test, y_test, act_map, feat_names):
    print("\n" + "="*70)
    print("LightGBM 模型训练")
    print("="*70)

    y_train_0 = y_train - 1
    y_test_0 = y_test - 1

    configs = [
        {"name":"LGBM_Light","n_estimators":50,"lr":0.12,"num_leaves":15,"max_depth":4,"min_child":25,"subsample":0.80,"colsample":0.70,"reg_lambda":1.5,"reg_alpha":0.0},
        {"name":"LGBM_Balanced","n_estimators":75,"lr":0.10,"num_leaves":23,"max_depth":5,"min_child":20,"subsample":0.85,"colsample":0.80,"reg_lambda":1.0,"reg_alpha":0.0},
        {"name":"LGBM_Deep","n_estimators":100,"lr":0.08,"num_leaves":31,"max_depth":6,"min_child":15,"subsample":0.85,"colsample":0.80,"reg_lambda":1.5,"reg_alpha":0.0},
    ]

    fit_idx, val_idx = get_split(X_train, y_train_0, subj_train)
    X_fit, y_fit = X_train[fit_idx], y_train_0[fit_idx]
    X_val, y_val = X_train[val_idx], y_train_0[val_idx]

    best_model, best_cfg, best_f1 = None, None, -1
    results = []

    for cfg in configs:
        print(f"\n训练 {cfg['name']}...")
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=NUM_CLASSES, boosting_type="gbdt",
            n_estimators=cfg["n_estimators"], learning_rate=cfg["lr"],
            num_leaves=cfg["num_leaves"], max_depth=cfg["max_depth"],
            min_child_samples=cfg["min_child"], subsample=cfg["subsample"],
            subsample_freq=1, colsample_bytree=cfg["colsample"],
            reg_lambda=cfg["reg_lambda"], reg_alpha=cfg["reg_alpha"],
            n_jobs=N_JOBS, random_state=RANDOM_STATE, verbosity=-1, force_col_wise=True)

        t0 = time.time()
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                  eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(10, verbose=False)])
        t = time.time() - t0

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)
        m = calc_metrics(y_val, y_pred, y_prob)
        m["train_time"] = t; m["model"] = cfg["name"]
        results.append(m)
        print(f"  Val Accuracy={m['accuracy']:.4f}, Macro-F1={m['macro_f1']:.4f}, Time={t:.1f}s")

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]; best_model = model; best_cfg = cfg

    # 最终训练
    n_rounds = getattr(best_model, 'best_iteration_', best_cfg["n_estimators"])
    if n_rounds is None or n_rounds <= 0: n_rounds = best_cfg["n_estimators"]
    n_rounds = max(1, min(n_rounds, best_cfg["n_estimators"]))

    final_model = lgb.LGBMClassifier(
        objective="multiclass", num_class=NUM_CLASSES, boosting_type="gbdt",
        n_estimators=n_rounds, learning_rate=best_cfg["lr"],
        num_leaves=best_cfg["num_leaves"], max_depth=best_cfg["max_depth"],
        min_child_samples=best_cfg["min_child"], subsample=best_cfg["subsample"],
        subsample_freq=1, colsample_bytree=best_cfg["colsample"],
        reg_lambda=best_cfg["reg_lambda"], reg_alpha=best_cfg["reg_alpha"],
        n_jobs=N_JOBS, random_state=RANDOM_STATE, verbosity=-1, force_col_wise=True)

    t0 = time.time()
    final_model.fit(X_train, y_train_0)
    final_time = time.time() - t0

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)
    test_m = calc_metrics(y_test_0, y_pred, y_prob)
    test_m["train_time"] = final_time; test_m["model"] = "LightGBM"

    # 混淆矩阵
    act_map_0 = {i: act_map[i+1] for i in range(6)}
    names = [act_map_0[i] for i in range(6)]
    cm = confusion_matrix(y_test_0, y_pred)
    fig, ax = plt.subplots(figsize=(9,7))
    im = ax.imshow(cm/cm.sum(axis=1,keepdims=True), cmap='Blues', aspect='auto')
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(names, rotation=30, ha='right'); ax.set_yticklabels(names)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{cm[i,j]/cm.sum(axis=1)[i]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title("LightGBM - Normalized Confusion Matrix")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "lightgbm_confusion_matrix.png", dpi=300); plt.close()

    # 特征重要性
    imp = pd.DataFrame({"feature": feat_names, "importance": final_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.barh(range(20), imp["importance"].values[::-1], color=C[2], edgecolor='white')
    ax.set_yticks(range(20)); ax.set_yticklabels(imp["feature"].values[::-1], fontsize=7)
    ax.set_xlabel("Importance"); ax.set_title("LightGBM Top-20 Feature Importance")
    apply_style(ax); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "lightgbm_feature_importance.png", dpi=300); plt.close()

    joblib.dump(final_model, MODEL_DIR / "lightgbm_model.joblib")
    print(f"LightGBM final - Test Accuracy={test_m['accuracy']:.4f}, Macro-F1={test_m['macro_f1']:.4f}, Time={final_time:.1f}s")

    return test_m, results


# ============================================================
# 主程序
# ============================================================
def main():
    start = time.time()
    print("="*70)
    print("实验三 Step 2: 传统机器学习模型 (GBDT, XGBoost, LightGBM)")
    print("="*70)

    X_train, X_test, y_train, y_test, subj_train, subj_test, feat_names, y_train_0 = load_data()
    act_map = load_activity_map()
    print(f"数据: train={X_train.shape}, test={X_test.shape}")

    all_test_results = []

    # GBDT
    gbdt_test, gbdt_val = train_gbdt(X_train, y_train, subj_train, X_test, y_test, act_map, feat_names)
    all_test_results.append(gbdt_test)

    # XGBoost
    xgb_test, xgb_val = train_xgboost(X_train, y_train, subj_train, X_test, y_test, act_map, feat_names)
    all_test_results.append(xgb_test)

    # LightGBM
    lgb_test, lgb_val = train_lightgbm(X_train, y_train, subj_train, X_test, y_test, act_map, feat_names)
    all_test_results.append(lgb_test)

    # 保存所有结果
    df = pd.DataFrame(all_test_results)
    df.to_csv(RESULTS_DIR / "traditional_ml_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n传统ML结果已保存到 {RESULTS_DIR / 'traditional_ml_results.csv'}")
    print(df[["model","accuracy","macro_f1","weighted_f1","train_time"]].to_string(index=False))

    elapsed = time.time() - start
    print(f"\nStep 2 完成! 总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
