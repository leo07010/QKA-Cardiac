#!/usr/bin/env python3
"""
完整論文實驗套件
================
執行所有論文所需的實驗

實驗清單:
1. Qubit Scaling: 10/20/30 qubits
2. Layer Depth: 1/2/4 layers
3. Benchmark: QKA vs Classical

使用方式:
    python run_all_experiments.py --quick    # 快速測試 (50 iter)
    python run_all_experiments.py --full     # 完整實驗 (300 iter)
"""
import numpy as np
import pandas as pd
import json
import time
import argparse
from datetime import datetime

from run_main import (
    load_data, optimize_qka, compute_kernel,
    evaluate_loocv, evaluate_rbf, get_n_params, KERNEL_MAP
)
from config import RESULT_DATA_DIR, RESULT_KERNEL_DIR, RESULT_FIG_DIR

# ==========================================
# 實驗 1: Qubit Scaling
# ==========================================
def exp_qubit_scaling(max_iter=50, shots=2000):
    """比較 10/20/30 qubits"""
    print("\n" + "="*60)
    print("Experiment 1: Qubit Scaling")
    print("="*60)

    configs = [10, 20, 30]
    n_layers = 2
    results = []

    for pca_dim in configs:
        print(f"\n>>> {pca_dim} qubits...")

        X, y, scaler_y, _, metadata = load_data('top30', pca_dim)

        # QKA
        start = time.time()
        opt_params, loss_history, train_time = optimize_qka(X, y, n_layers, max_iter, shots)
        K = compute_kernel(X, opt_params, n_layers, shots)
        qka = evaluate_loocv(K, y, scaler_y)

        # RBF
        rbf_start = time.time()
        rbf = evaluate_rbf(X, y, scaler_y)
        rbf_time = time.time() - rbf_start

        results.append({
            'qubits': pca_dim,
            'n_params': get_n_params(pca_dim, n_layers),
            'qka_r2': qka['r2'], 'qka_mae': qka['mae'], 'qka_time': train_time,
            'rbf_r2': rbf['r2'], 'rbf_mae': rbf['mae'], 'rbf_time': rbf_time
        })

        np.save(f"{RESULT_KERNEL_DIR}/qubit_{pca_dim}_K.npy", K)

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULT_DATA_DIR}/exp_qubit_scaling.csv", index=False)
    print(f"\nSaved: exp_qubit_scaling.csv")
    print(df.to_string(index=False))
    return results

# ==========================================
# 實驗 2: Layer Depth
# ==========================================
def exp_layer_depth(max_iter=50, shots=2000):
    """比較 1/2/4 layers"""
    print("\n" + "="*60)
    print("Experiment 2: Layer Depth")
    print("="*60)

    configs = [1, 2, 4]
    pca_dim = 10
    results = []

    X, y, scaler_y, _, _ = load_data('top30', pca_dim)

    for n_layers in configs:
        print(f"\n>>> {n_layers} layer(s)...")

        opt_params, loss_history, train_time = optimize_qka(X, y, n_layers, max_iter, shots)
        K = compute_kernel(X, opt_params, n_layers, shots)
        qka = evaluate_loocv(K, y, scaler_y)

        results.append({
            'layers': n_layers,
            'n_params': get_n_params(pca_dim, n_layers),
            'r2': qka['r2'], 'mae': qka['mae'],
            'kta': -loss_history[-1], 'time': train_time
        })

        np.save(f"{RESULT_KERNEL_DIR}/layer_{n_layers}_K.npy", K)

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULT_DATA_DIR}/exp_layer_depth.csv", index=False)
    print(f"\nSaved: exp_layer_depth.csv")
    print(df.to_string(index=False))
    return results

# ==========================================
# 實驗 3: Benchmark
# ==========================================
def exp_benchmark(max_iter=50, shots=2000):
    """QKA vs Classical Models"""
    print("\n" + "="*60)
    print("Experiment 3: Benchmark")
    print("="*60)

    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score, mean_absolute_error

    pca_dim = 10
    n_layers = 2

    X, y, scaler_y, _, _ = load_data('top30', pca_dim)

    # Train QKA
    print("\n>>> Training QKA...")
    opt_params, loss_history, train_time = optimize_qka(X, y, n_layers, max_iter, shots)
    K = compute_kernel(X, opt_params, n_layers, shots)

    models = {
        'QKA-SVR': ('precomputed', K, train_time),
        'RBF-SVR': ('rbf', None, 0),
        'Linear-SVR': ('linear', None, 0),
        'RandomForest': ('rf', None, 0),
    }

    results = []

    for name, (ktype, K_pre, base_time) in models.items():
        print(f"\n>>> Evaluating {name}...")

        loo = LeaveOneOut()
        y_true_list, y_pred_list = [], []
        start = time.time()

        for train_idx, test_idx in loo.split(X):
            if ktype == 'precomputed':
                model = SVR(kernel='precomputed', C=100, epsilon=0.1)
                model.fit(K_pre[np.ix_(train_idx, train_idx)], y[train_idx])
                pred = model.predict(K_pre[np.ix_(test_idx, train_idx)])[0]
            elif ktype == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])[0]
            else:
                model = SVR(kernel=ktype, C=100, epsilon=0.1)
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])[0]

            true_val = scaler_y.inverse_transform([[y[test_idx][0]]])[0][0]
            pred_val = scaler_y.inverse_transform([[pred]])[0][0]
            y_true_list.append(true_val)
            y_pred_list.append(pred_val)

        eval_time = time.time() - start

        results.append({
            'model': name,
            'r2': r2_score(y_true_list, y_pred_list),
            'mae': mean_absolute_error(y_true_list, y_pred_list),
            'time': base_time if base_time > 0 else eval_time
        })

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULT_DATA_DIR}/exp_benchmark.csv", index=False)
    print(f"\nSaved: exp_benchmark.csv")
    print(df.to_string(index=False))
    return results

# ==========================================
# 主函數
# ==========================================
def run_all(quick=True):
    """執行所有實驗"""
    max_iter = 50 if quick else 300
    shots = 2000

    print("\n" + "="*70)
    print("  QKA Complete Experiment Suite")
    print("="*70)
    print(f"  Mode: {'Quick' if quick else 'Full'}")
    print(f"  Max iterations: {max_iter}")
    print("="*70)

    all_results = {}
    all_results['qubit_scaling'] = exp_qubit_scaling(max_iter, shots)
    all_results['layer_depth'] = exp_layer_depth(max_iter, shots)
    all_results['benchmark'] = exp_benchmark(max_iter, shots)

    # 儲存總結
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{RESULT_DATA_DIR}/all_results_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("  ALL EXPERIMENTS COMPLETED!")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--quick', action='store_true', help='Quick mode (50 iter)')
    parser.add_argument('--full', action='store_true', help='Full mode (300 iter)')
    args = parser.parse_args()

    run_all(quick=not args.full)

if __name__ == "__main__":
    main()
