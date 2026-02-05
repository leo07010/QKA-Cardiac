# Quantum-Genomic-Kernel-Alignment - Quantum Machine Learning for Genomics

這個專案旨在比較量子機器學習（QML）與古典機器學習方法在基因表達數據上的回歸預測性能。我們特別關注 **Quantum Kernel Alignment (QKA)** 結合 **SVR (QKA-QSVR)** 的應用。

專案目前分為兩個主要的實作環境：
1.  **CUDAQ**: 使用 NVIDIA CUDA-Q 框架。
2.  **Qiskit**: 使用 IBM Qiskit 框架。

## 📁 專案結構

```
Genenet_new/
├── CUDAQ/
│   └── QKA_Test.ipynb        # CUDA-Q 實作筆記本
├── Qiskit/
│   ├── QKA_Test_Qiskit.ipynb # Qiskit 實作筆記本
│   └── data/                 # 測試所需的基因表達數據
├── results/                  # 實驗結果輸出目錄
├── requirements.txt          # Python 依賴套件
└── README.md                 # 本文件
```

## 🚀 快速開始

### 1. 環境準備

請確保您已安裝 Python 3.10+，並建議使用虛擬環境。

```bash
# 安裝基本依賴
pip install -r requirements.txt

# 根據您要執行的環境安裝額外套件：

# 若使用 CUDAQ (需要 NVIDIA GPU):
# pip install cudaq

# 若使用 Qiskit:
# pip install qiskit qiskit-machine-learning qiskit-algorithms
```

### 2. 執行測試

#### A. 測試 CUDA-Q 版本
開啟並執行 `CUDAQ/QKA_Test.ipynb`。此筆記本包含使用 CUDA-Q 進行量子核計算與 SVR 訓練的完整流程。

#### B. 測試 Qiskit 版本
開啟並執行 `Qiskit/QKA_Test_Qiskit.ipynb`。此筆記本演示了如何使用 Qiskit 的 `QuantumKernelTrainer` 進行量子核對齊，並結合 `QSVR` 進行預測。

## 🎯 核心功能與方法

本專案主要比較以下模型的性能：
-   **QKA-QSVR (Quantum Kernel Alignment - Quantum Support Vector Regression)**: 使用參數化量子電路作為核函數，並透過優化變分參數來對齊核矩陣，最後用於 SVR。
-   **RBF-SVR**: 傳統的使用徑向基函數 (RBF) 核的 SVR。
-   **GLM**: 廣義線性模型 (Ridge Regression) 作為基準。

實驗流程通常包含：
1.  **數據預處理**: 標準化 (StandardScaler) 與 降維 (PCA)。
2.  **量子特徵映射**: 將古典數據編碼至量子態 (例如使用 ZZFeatureMap)。
3.  **核矩陣計算**: 計算訓練數據與測試數據的核矩陣。
4.  **模型訓練與評估**: 使用 LOOCV (Leave-One-Out Cross-Validation) 評估 R2 分數與 RMSE。



## 📊 預期結果

執行筆記本後，您將看到：
-   每個 Hold-out 測試集的 RMSE 與 R² 分數。
-   整體 LOOCV 的平均性能指標。
-   Qiskit 版本會顯示量子核訓練過程中的 Loss 變化。

## 🛠️ 常見問題

-   **CUDA-Q 錯誤**: 請確認您的機器配備 NVIDIA GPU 且已安裝正確版本的 CUDA 驅動程式。若無 GPU，請確保 `cudaq` 設置為 CPU 模式 (`cudaq.set_target("qpp-cpu")`)。
-   **記憶體不足**: 量子模擬（特別是 Statevector 模擬）非常消耗記憶體。若遇到問題，請嘗試減少 PCA 的主成分數量（即減少 Qubits 數）或減少數據量。

## 📝 參考

本專案參考了 `GPU_QKA.py` 的核心邏輯，並分別移植到 CUDA-Q 與 Qiskit 框架以進行比較測試。
