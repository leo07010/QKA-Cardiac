# QKA-Cardiac: Quantum Kernel Alignment for Cardiac Genomics

> 使用量子機器學習方法進行心臟基因表達數據預測分析

## 📌 項目簡介

本項目實現了 **量子核對齊 (Quantum Kernel Alignment, QKA)** 方法，並將其應用於心臟基因表達數據的回歸預測任務。通過與古典機器學習方法的系統比較，驗證量子方法在生物信息學中的潛力。

### 🎯 核心特點
- ✅ **CUDA-Q 實現**: 使用 NVIDIA CUDA-Q 框架進行高效量子計算
- ✅ **完整實驗套件**: Qubit 可擴展性、層深度、基準比較等
- ✅ **並行執行**: 支持多 GPU 同時運行不同實驗
- ✅ **論文級結果**: 完整的數據分析、圖表生成和報告

---

## 📁 項目結構

```
QKA-Cardiac/
├── CUDAQ/                          # 主要實驗環境 (CUDA-Q)
│   ├── run_main.py                 # 單次實驗執行
│   ├── run_all_experiments.py      # 完整實驗套件
│   ├── run_final_comparison.py     # Final comparison (1000 iter)
│   ├── config.py                   # 配置文件
│   ├── analyze_quantum_correlations.py # 量子相關性分析
│   ├── generate_figures.py         # 圖表生成
│   ├── cleanup.sh                  # 清理腳本
│   ├── README.md                   # CUDAQ 詳細說明
│   └── results/                    # 實驗結果
│       ├── data/                   # CSV, JSON 結果
│       ├── figures/                # 圖表輸出
│       ├── kernels/                # 核矩陣 (NPY)
│       └── logs/                   # 實驗日誌
│
├── Qiskit/                         # 備選實現 (IBM Qiskit)
│   ├── QKA_Test_Qiskit.ipynb      # Qiskit 筆記本
│   └── ...
│
├── data/                           # 數據文件
│   ├── cardiac_formatted_dataset-001.csv
│   ├── relevancy_ranked_genes_top_30.csv
│   ├── relevancy_ranked_genes_top_50.csv
│   └── relevancy_ranked_genes_top_100.csv
│
├── docs/                           # 文檔和論文
├── bin/                            # 輔助工具
├── run_experiments.sh              # GPU 0 後台執行腳本
├── run_all_experiments_gpu1.sh     # GPU 1 後台執行腳本
├── monitor_experiment.sh           # 監控面板
├── requirements.txt                # Python 依賴
└── README.md                       # 主說明文件
```

---

## 🚀 快速開始

### 1️⃣ 環境激活

```bash
# 激活 conda 環境
conda activate TN-VQE

# 驗證 CUDA-Q
python -c "import cudaq; print(cudaq.__version__)"
```

### 2️⃣ 簡單實驗（5 分鐘）

```bash
cd CUDAQ

# 快速測試 (50 iterations)
python run_main.py --pca_dim 10 --n_layers 2 --max_iter 50
```

### 3️⃣ 完整論文實驗

```bash
# 完整模式 (300 iterations)
python run_all_experiments.py --full

# 或指定 GPU 1
CUDA_VISIBLE_DEVICES=1 python run_all_experiments.py --full
```

### 4️⃣ 後台執行（推薦用於長時間實驗）

```bash
# GPU 0: final comparison (1000 iterations)
nohup bash run_experiments.sh > nohup.out 2>&1 &

# GPU 1: 全部實驗
nohup bash run_all_experiments_gpu1.sh > nohup_gpu1.out 2>&1 &

# 監控進度
bash monitor_experiment.sh
```

### 5️⃣ 查看結果

```bash
# 檢視 CSV 結果
head CUDAQ/results/data/exp_qubit_scaling.csv

# 生成圖表
python CUDAQ/generate_figures.py
```

---

## 🔬 實驗范圍

### 實驗 1: Qubit 可擴展性
- 量子位元數: 10, 20 qubits
- 層數: 2 層
- 指標: R², MAE, 訓練時間

### 實驗 2: 電路層深度
- 層數: 1, 2, 4 層
- 量子位元數: 10 qubits
- 指標: R², KTA, 參數量

### 實驗 3: 基準比較
- QKA-SVR vs RBF-SVR vs Linear-SVR vs RandomForest
- 30 個基因特徵
- LOOCV 評估

---

## 📊 預期結果

成功完成實驗後，您將獲得：

```
CUDAQ/results/
├── data/
│   ├── exp_qubit_scaling.csv       # Qubit 實驗: R²、MAE、時間
│   ├── exp_layer_depth.csv         # 層數實驗: R²、KTA、參數量
│   ├── exp_benchmark.csv           # 基準對比: 各模型性能
│   ├── final_30q_comparison.csv    # Final comparison 結果
│   └── all_results_*.json          # 實驗詳細數據
│
└── figures/
    ├── fig_01_kernel_matrix.png    # 核矩陣熱圖
    ├── fig_02_qubit_scaling.png    # Qubit 性能曲線
    ├── fig_03_layer_depth.png      # 層深度影響
    ├── fig_04_benchmark.png        # 模型對比柱狀圖
    ├── fig_05_convergence.png      # KTA 收斂曲線
    ├── fig_06_prediction.png       # 預測準確性散點圖
    ├── fig_07_weights.png          # 參數敏感性
    └── fig_08_correlation_heatmap.png # 特徵相關性
```

---

## 📚 詳細文檔

- **[CUDAQ 實現詳情](CUDAQ/README.md)** - 完整的參數、命令和實驗指南
- **[Qiskit 筆記本](Qiskit/QKA_Test_Qiskit.ipynb)** - Qiskit 框架實現

---

## 🛠️ 常見問題

| 問題 | 解決方案 |
|------|--------|
| `ModuleNotFoundError: cudaq` | 確保環境激活：`conda activate TN-VQE` |
| CUDA 記憶體不足 | 降低 `--shots` 或 `--pca_dim` 參數 |
| 進程卡住 | 檢查 `nvidia-smi`，查看是否有其他進程占用 GPU |
| 結果文件缺失 | 檢查 `CUDAQ/results/` 目錄權限和磁盤空間 |

---

## 📬 聯絡方式

**作者**: Leo Chen  
**Email**: leo07010@gmail.com  
**項目日期**: 2026年2月

---

## 📄 授權

本項目採用 MIT 授權。詳見 [LICENSE](LICENSE) 文件。

---

## 🙏 致謝

- NVIDIA CUDA-Q 框架開發團隊
- IBM Qiskit 社區
- 心臟基因表達數據貢獻者

---

**最後更新**: 2026-02-06  
**版本**: 1.0.0  
**狀態**: ✓ 活躍開發中
