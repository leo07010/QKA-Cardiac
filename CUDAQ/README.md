# QKA: Quantum Kernel Alignment for Cardiac Gene Expression

## 📖 研究概述

本專案實作 **量子核對齊 (Quantum Kernel Alignment, QKA)** 方法，用於心臟基因表達數據的預測分析。

---

## 📁 檔案結構

```
CUDAQ/
├── config.py               # 配置文件 (路徑、參數)
├── run_main.py             # 主程式 (單次實驗)
├── run_all_experiments.py  # 完整實驗套件
├── generate_figures.py     # 論文圖表生成
├── cleanup.sh              # 清理舊檔案腳本
├── README.md               # 說明文件
│
├── data/                   # 數據資料夾
│   ├── cardiac_formatted_dataset-001.csv     # 主數據
│   ├── relevancy_ranked_genes_top_30.csv     # Top 30 基因
│   ├── relevancy_ranked_genes_top_50.csv     # Top 50 基因
│   └── relevancy_ranked_genes_top_100.csv    # Top 100 基因
│
└── results/                # 實驗結果
    ├── data/               # CSV, JSON 結果
    ├── figures/            # 圖表 PNG
    └── kernels/            # 核矩陣 NPY
```

---

## 🚀 快速開始

### 1. 清理舊檔案 (首次使用)

```bash
bash cleanup.sh
```

### 2. 執行單次實驗

```bash
# 快速測試 (50 iterations, ~5分鐘)
python run_main.py --pca_dim 10 --n_layers 2 --max_iter 50

# 完整實驗 (300 iterations)
python run_main.py --pca_dim 20 --n_layers 2 --max_iter 300 --gene_set top50
```

### 3. 執行完整論文實驗

```bash
# 快速模式
python run_all_experiments.py --quick

# 完整模式
python run_all_experiments.py --full
```

### 4. 生成圖表

```bash
python generate_figures.py
```

---

## 📊 論文圖表

| 圖號 | 名稱 | 檔名 |
|------|------|------|
| 1 | Kernel Matrix | `fig_01_kernel_matrix.png` |
| 2 | Qubit Scaling | `fig_02_qubit_scaling.png` |
| 3 | Layer Depth | `fig_03_layer_depth.png` |
| 4 | Benchmark | `fig_04_benchmark.png` |
| 5 | Convergence | `fig_05_convergence.png` |
| 6 | Prediction | `fig_06_prediction.png` |
| 7 | Weights | `fig_07_weights.png` |

---

## 🔬 實驗設計

### 實驗 1: Qubit Scaling
- **配置**: 10, 20, 30 qubits (PCA 維度)
- **指標**: R², MAE, 訓練時間

### 實驗 2: Layer Depth
- **配置**: 1, 2, 4 layers
- **指標**: R², KTA, 參數量

### 實驗 3: Benchmark
- **模型**: QKA-SVR, RBF-SVR, Linear-SVR, RandomForest

---

## 🧬 量子電路

```
q₀: ─H─RZ(x₀·θ₀)─●───────RX(x₀·θ₄)─...─M
                 │
q₁: ─H─RZ(x₁·θ₁)─X──●────RX(x₁·θ₅)─...─M
                    │
q₂: ─H─RZ(x₂·θ₂)────X──●─RX(x₂·θ₆)─...─M
                       │
q₃: ─H─RZ(x₃·θ₃)───────X─RX(x₃·θ₇)─...─M
```

**核計算**: K(x, x') = |⟨0|U†(x')U(x)|0⟩|²

**參數數量**: n_qubits × n_layers × 2

---

## 📝 命令列參數

```bash
python run_main.py [OPTIONS]

Options:
  --pca_dim INT      PCA 維度 (量子位元數) [default: 10]
  --n_layers INT     電路層數 (1, 2, 4) [default: 2]
  --max_iter INT     最大迭代次數 [default: 50]
  --shots INT        量子採樣次數 [default: 2000]
  --gene_set STR     基因集 (top30, top50, top100) [default: top30]
```

---

## 📬 聯繫

- Email: leo07010@gmail.com
