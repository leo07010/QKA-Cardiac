#!/usr/bin/env python3
"""
QKA 主程式
==========
心臟基因表達量子核對齊 (Quantum Kernel Alignment)

使用方式:
    python run_main.py --pca_dim 10 --n_layers 2 --max_iter 50 --gene_set top30
    python run_main.py --pca_dim 20 --n_layers 2 --max_iter 300 --gene_set top50
"""
import cudaq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
import os
import time
import argparse
import json
from datetime import datetime

# 導入配置
from config import (
    MAIN_DATA_FILE, GENE_FILES, RESULT_DIR,
    RESULT_DATA_DIR, RESULT_FIG_DIR, RESULT_KERNEL_DIR,
    DEFAULT_SHOTS, DEFAULT_MAX_ITER, SVR_C, SVR_EPSILON
)

# ==========================================
# CUDA-Q Target 設定
# ==========================================
try:
    cudaq.set_target("nvidia")
    CUDAQ_TARGET = "nvidia-gpu"
except:
    try:
        cudaq.set_target("qpp-cpu")
        CUDAQ_TARGET = "qpp-cpu"
    except:
        CUDAQ_TARGET = "default"

print(f"[System] CUDA-Q Target: {CUDAQ_TARGET}")

# ==========================================
# 量子電路定義 (Angle Encoding)
# ==========================================

# --- 1 Layer ---
@cudaq.kernel
def angle_map_L1(q: cudaq.qview, x: list[float], params: list[float]):
    n = x.size()
    for i in range(n):
        h(q[i])
        rz(x[i] * params[i], q[i])
    for i in range(n - 1):
        cx(q[i], q[i+1])
    if n > 1:
        cx(q[n-1], q[0])
    for i in range(n):
        rx(x[i] * params[n + i], q[i])

# --- 2 Layers ---
@cudaq.kernel
def angle_map_L2(q: cudaq.qview, x: list[float], params: list[float]):
    n = x.size()
    # Layer 1
    for i in range(n):
        h(q[i])
        rz(x[i] * params[i], q[i])
    for i in range(n - 1):
        cx(q[i], q[i+1])
    if n > 1:
        cx(q[n-1], q[0])
    for i in range(n):
        rx(x[i] * params[n + i], q[i])
    # Layer 2
    base = n * 2
    for i in range(n):
        rz(x[i] * params[base + i], q[i])
    for i in range(n - 1):
        cx(q[i], q[i+1])
    if n > 1:
        cx(q[n-1], q[0])
    for i in range(n):
        rx(x[i] * params[base + n + i], q[i])

# --- 4 Layers ---
@cudaq.kernel
def angle_map_L4(q: cudaq.qview, x: list[float], params: list[float]):
    n = x.size()
    # Layer 1
    for i in range(n):
        h(q[i])
        rz(x[i] * params[i], q[i])
    for i in range(n - 1):
        cx(q[i], q[i+1])
    if n > 1:
        cx(q[n-1], q[0])
    for i in range(n):
        rx(x[i] * params[n + i], q[i])
    # Layer 2
    base = n * 2
    for i in range(n):
        rz(x[i] * params[base + i], q[i])
    for i in range(n - 1):
        cx(q[i], q[i+1])
    if n > 1:
        cx(q[n-1], q[0])
    for i in range(n):
        rx(x[i] * params[base + n + i], q[i])
    # Layer 3
    base = n * 4
    for i in range(n):
        rz(x[i] * params[base + i], q[i])
    for i in range(n - 1):
        cx(q[i], q[i+1])
    if n > 1:
        cx(q[n-1], q[0])
    for i in range(n):
        rx(x[i] * params[base + n + i], q[i])
    # Layer 4
    base = n * 6
    for i in range(n):
        rz(x[i] * params[base + i], q[i])
    for i in range(n - 1):
        cx(q[i], q[i+1])
    if n > 1:
        cx(q[n-1], q[0])
    for i in range(n):
        rx(x[i] * params[base + n + i], q[i])

# --- Kernel Entry Points ---
@cudaq.kernel
def kernel_L1(x1: list[float], x2: list[float], params: list[float]):
    q = cudaq.qvector(len(x1))
    angle_map_L1(q, x1, params)
    cudaq.adjoint(angle_map_L1, q, x2, params)
    mz(q)

@cudaq.kernel
def kernel_L2(x1: list[float], x2: list[float], params: list[float]):
    q = cudaq.qvector(len(x1))
    angle_map_L2(q, x1, params)
    cudaq.adjoint(angle_map_L2, q, x2, params)
    mz(q)

@cudaq.kernel
def kernel_L4(x1: list[float], x2: list[float], params: list[float]):
    q = cudaq.qvector(len(x1))
    angle_map_L4(q, x1, params)
    cudaq.adjoint(angle_map_L4, q, x2, params)
    mz(q)

KERNEL_MAP = {1: kernel_L1, 2: kernel_L2, 4: kernel_L4}

def get_n_params(n_qubits: int, n_layers: int) -> int:
    """參數數量 = n_qubits × n_layers × 2"""
    return n_qubits * n_layers * 2

# ==========================================
# 數據載入
# ==========================================
def load_data(gene_set: str = 'top30', pca_dim: int = None):
    """
    載入並預處理數據

    Args:
        gene_set: 基因集 ('top30', 'top50', 'top100')
        pca_dim: PCA 維度 (None = 不降維)

    Returns:
        X, y, scaler_y, feature_names, metadata
    """
    print(f"\n{'='*50}")
    print(f"Loading Data: {gene_set}")
    print(f"{'='*50}")

    gene_file = GENE_FILES.get(gene_set, GENE_FILES['top30'])

    # 讀取數據
    df_main = pd.read_csv(MAIN_DATA_FILE)
    df_gene = pd.read_csv(gene_file)

    # 基因欄位
    gene_col = 'symbol'
    target_genes = df_gene[gene_col].tolist()
    valid_genes = [g for g in target_genes if g in df_main.columns]

    print(f"   Target genes: {len(target_genes)}")
    print(f"   Matched genes: {len(valid_genes)}")

    # 提取數據
    y_raw = df_main['Beat count per min'].values
    X_raw = df_main[valid_genes].values

    n_samples = X_raw.shape[0]
    n_features_orig = X_raw.shape[1]

    print(f"   Samples: {n_samples}")
    print(f"   Original features: {n_features_orig}")

    # 標準化
    X_std = StandardScaler().fit_transform(X_raw)

    # PCA
    feature_names = valid_genes
    pca_variance = None

    if pca_dim and n_features_orig > pca_dim:
        pca = PCA(n_components=pca_dim)
        X_reduced = pca.fit_transform(X_std)
        pca_variance = sum(pca.explained_variance_ratio_) * 100
        feature_names = [f"PC{i+1}" for i in range(pca_dim)]
        print(f"   PCA: {n_features_orig} -> {pca_dim} ({pca_variance:.1f}% variance)")
    else:
        X_reduced = X_std
        pca_dim = n_features_orig

    # 量子縮放 [0.1, π+0.1]
    scaler_x = MinMaxScaler(feature_range=(0.1, np.pi + 0.1))
    X = scaler_x.fit_transform(X_reduced)

    # 目標標準化
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    print(f"   Final X shape: {X.shape}")

    metadata = {
        'n_samples': n_samples,
        'n_features_original': n_features_orig,
        'n_features_pca': pca_dim,
        'pca_variance': pca_variance,
        'gene_set': gene_set
    }

    return X, y, scaler_y, feature_names, metadata

# ==========================================
# 量子核計算
# ==========================================
def compute_kernel(X: np.ndarray, params: np.ndarray,
                   n_layers: int, shots: int = 2000) -> np.ndarray:
    """計算量子核矩陣"""
    n_samples = len(X)
    n_qubits = X.shape[1]
    K = np.zeros((n_samples, n_samples))
    zero_str = '0' * n_qubits

    kernel_func = KERNEL_MAP[n_layers]
    total = n_samples * (n_samples + 1) // 2
    count = 0

    for i in range(n_samples):
        for j in range(i, n_samples):
            result = cudaq.sample(
                kernel_func,
                X[i].tolist(), X[j].tolist(), params.tolist(),
                shots_count=shots
            )
            prob = dict(result.items()).get(zero_str, 0) / shots
            K[i, j] = prob
            K[j, i] = prob

            count += 1
            if count % 50 == 0:
                print(f"\r   Kernel: {count}/{total} ({100*count/total:.0f}%)", end="")

    print(f"\r   Kernel: {total}/{total} (100%) Done!     ")
    return K

# ==========================================
# QKA 優化
# ==========================================
def optimize_qka(X: np.ndarray, y: np.ndarray, n_layers: int,
                 max_iter: int = 200, shots: int = 2000):
    """優化量子核參數"""
    n_qubits = X.shape[1]
    n_params = get_n_params(n_qubits, n_layers)

    print(f"\n{'='*50}")
    print(f"QKA Optimization")
    print(f"{'='*50}")
    print(f"   Qubits: {n_qubits}, Layers: {n_layers}")
    print(f"   Parameters: {n_params}, MaxIter: {max_iter}")

    loss_history = []

    def loss_kta(params):
        params = np.abs(params)
        K = compute_kernel(X, params, n_layers, shots)

        Y_mat = np.outer(y, y)
        inner = np.sum(K * Y_mat)
        norm_K = np.linalg.norm(K, 'fro') + 1e-9
        norm_Y = np.linalg.norm(Y_mat, 'fro')
        kta = inner / (norm_K * norm_Y)

        loss = -kta
        loss_history.append(loss)
        print(f"\r   Iter {len(loss_history):3d}: KTA = {-loss:.4f}", end="")
        return loss

    np.random.seed(42)
    init_params = np.random.uniform(0.1, 2*np.pi, n_params)

    print(f"\n   Starting optimization...")
    start = time.time()

    result = minimize(loss_kta, init_params, method='COBYLA',
                     options={'maxiter': max_iter, 'rhobeg': 0.5})

    train_time = time.time() - start
    opt_params = np.abs(result.x)

    print(f"\n   Done! Final KTA: {-result.fun:.4f}")
    print(f"   Time: {train_time:.1f}s")

    return opt_params, loss_history, train_time

# ==========================================
# 評估
# ==========================================
def evaluate_loocv(K: np.ndarray, y: np.ndarray, scaler_y):
    """LOOCV 評估"""
    loo = LeaveOneOut()
    y_true_list, y_pred_list = [], []

    for train_idx, test_idx in loo.split(K):
        svr = SVR(kernel='precomputed', C=SVR_C, epsilon=SVR_EPSILON)
        svr.fit(K[np.ix_(train_idx, train_idx)], y[train_idx])
        pred = svr.predict(K[np.ix_(test_idx, train_idx)])[0]

        true_val = scaler_y.inverse_transform([[y[test_idx][0]]])[0][0]
        pred_val = scaler_y.inverse_transform([[pred]])[0][0]

        y_true_list.append(true_val)
        y_pred_list.append(pred_val)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'y_true': y_true,
        'y_pred': y_pred
    }

def evaluate_rbf(X: np.ndarray, y: np.ndarray, scaler_y):
    """古典 RBF SVR"""
    loo = LeaveOneOut()
    y_true_list, y_pred_list = [], []

    for train_idx, test_idx in loo.split(X):
        svr = SVR(kernel='rbf', C=SVR_C, epsilon=SVR_EPSILON, gamma='scale')
        svr.fit(X[train_idx], y[train_idx])
        pred = svr.predict(X[test_idx])[0]

        true_val = scaler_y.inverse_transform([[y[test_idx][0]]])[0][0]
        pred_val = scaler_y.inverse_transform([[pred]])[0][0]

        y_true_list.append(true_val)
        y_pred_list.append(pred_val)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'y_true': y_true,
        'y_pred': y_pred
    }

# ==========================================
# 繪圖
# ==========================================
def plot_results(K_qka, results_qka, results_rbf, opt_params,
                 loss_history, feature_names, n_layers, exp_name):
    """生成結果圖表"""

    # 1. 核矩陣
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(K_qka, cmap='Purples', ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Transition Probability'})
    ax.set_title(f'Quantum Kernel Matrix ({exp_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FIG_DIR, f'{exp_name}_kernel.png'), dpi=300)
    plt.close()

    # 2. 預測散點圖
    fig, ax = plt.subplots(figsize=(8, 8))
    y_true = results_qka['y_true']
    y_pred_q = results_qka['y_pred']
    y_pred_r = results_rbf['y_pred']

    min_v, max_v = min(y_true.min(), y_pred_q.min()), max(y_true.max(), y_pred_q.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.5, lw=2)
    ax.scatter(y_true, y_pred_r, c='blue', alpha=0.6, s=80,
               label=f"RBF (R²={results_rbf['r2']:.3f})")
    ax.scatter(y_true, y_pred_q, c='purple', alpha=0.8, s=100, marker='^',
               label=f"QKA (R²={results_qka['r2']:.3f})")
    ax.set_xlabel('True Beat Count', fontsize=12)
    ax.set_ylabel('Predicted Beat Count', fontsize=12)
    ax.set_title('LOOCV Prediction', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FIG_DIR, f'{exp_name}_prediction.png'), dpi=300)
    plt.close()

    # 3. 收斂曲線
    fig, ax = plt.subplots(figsize=(10, 6))
    kta = [-l for l in loss_history]
    ax.plot(range(1, len(kta)+1), kta, 'o-', color='indigo', markersize=3)
    ax.axhline(y=kta[-1], color='red', ls='--', alpha=0.5, label=f'Final: {kta[-1]:.4f}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('KTA', fontsize=12)
    ax.set_title('Optimization Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FIG_DIR, f'{exp_name}_convergence.png'), dpi=300)
    plt.close()

    # 4. 權重熱圖
    n_qubits = len(feature_names)
    try:
        w_mat = opt_params.reshape(n_layers * 2, n_qubits)
        row_labels = []
        for l in range(1, n_layers + 1):
            row_labels.extend([f'L{l}_RZ', f'L{l}_RX'])

        fig, ax = plt.subplots(figsize=(12, max(4, n_layers * 2)))
        sns.heatmap(np.abs(w_mat), annot=True, fmt='.2f', cmap='RdYlGn',
                    xticklabels=feature_names, yticklabels=row_labels, ax=ax)
        ax.set_title('Learned Parameters', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_FIG_DIR, f'{exp_name}_weights.png'), dpi=300)
        plt.close()
    except:
        pass

    print(f"\n   Figures saved to: {RESULT_FIG_DIR}")

# ==========================================
# 主函數
# ==========================================
def run_experiment(pca_dim: int = 10, n_layers: int = 2,
                   max_iter: int = 50, shots: int = 2000,
                   gene_set: str = 'top30'):
    """執行完整實驗"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"QKA_{pca_dim}q_{n_layers}L_{gene_set}"

    print("\n" + "="*60)
    print(f"  Experiment: {exp_name}")
    print("="*60)

    # 1. 載入數據
    X, y, scaler_y, feat_names, metadata = load_data(gene_set, pca_dim)

    # 2. QKA 優化
    opt_params, loss_history, train_time = optimize_qka(
        X, y, n_layers, max_iter, shots
    )

    # 3. 計算最終核矩陣
    print(f"\n   Computing final kernel...")
    K_qka = compute_kernel(X, opt_params, n_layers, shots)

    # 4. 評估
    print(f"\n{'='*50}")
    print(f"Evaluation")
    print(f"{'='*50}")

    results_qka = evaluate_loocv(K_qka, y, scaler_y)
    results_rbf = evaluate_rbf(X, y, scaler_y)

    print(f"\n   QKA-SVR:  R²={results_qka['r2']:.4f}, MAE={results_qka['mae']:.2f}")
    print(f"   RBF-SVR:  R²={results_rbf['r2']:.4f}, MAE={results_rbf['mae']:.2f}")

    # 5. 繪圖
    plot_results(K_qka, results_qka, results_rbf, opt_params,
                 loss_history, feat_names, n_layers, exp_name)

    # 6. 儲存結果
    np.save(os.path.join(RESULT_KERNEL_DIR, f'{exp_name}_K.npy'), K_qka)
    np.save(os.path.join(RESULT_KERNEL_DIR, f'{exp_name}_params.npy'), opt_params)

    summary = {
        'experiment': exp_name,
        'timestamp': timestamp,
        'config': {
            'pca_dim': pca_dim, 'n_layers': n_layers,
            'n_params': get_n_params(pca_dim, n_layers),
            'max_iter': max_iter, 'shots': shots, 'gene_set': gene_set
        },
        'metadata': metadata,
        'qka': {'r2': float(results_qka['r2']), 'mae': float(results_qka['mae']),
                'kta': float(-loss_history[-1]), 'time': float(train_time)},
        'rbf': {'r2': float(results_rbf['r2']), 'mae': float(results_rbf['mae'])}
    }

    with open(os.path.join(RESULT_DATA_DIR, f'{exp_name}.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("  Experiment Complete!")
    print("="*60)

    return summary

# ==========================================
# CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='QKA Main Runner')
    parser.add_argument('--pca_dim', type=int, default=10, help='PCA dimensions (qubits)')
    parser.add_argument('--n_layers', type=int, default=2, choices=[1, 2, 4], help='Circuit layers')
    parser.add_argument('--max_iter', type=int, default=50, help='Max iterations')
    parser.add_argument('--shots', type=int, default=2000, help='Quantum shots')
    parser.add_argument('--gene_set', type=str, default='top30',
                        choices=['top30', 'top50', 'top100'], help='Gene set')

    args = parser.parse_args()

    run_experiment(
        pca_dim=args.pca_dim,
        n_layers=args.n_layers,
        max_iter=args.max_iter,
        shots=args.shots,
        gene_set=args.gene_set
    )

if __name__ == "__main__":
    main()
