# QKA: Quantum Kernel Alignment for Cardiac Gene Expression Analysis

## 📖 Project Overview

This project implements **Quantum Kernel Alignment (QKA)**, a hybrid quantum-classical machine learning method for predicting cardiac gene expression patterns. The approach leverages parameterized quantum circuits to construct task-specific kernel functions for support vector regression, demonstrating the potential of quantum computing in bioinformatics applications.

**Key Innovation**: Use quantum circuits to compute kernel matrices that automatically adapt to the structure of cardiac gene expression data, improving prediction performance over classical kernel methods.

---

## 📁 Project Structure

```
CUDAQ/
├── config.py                       # Configuration file (paths, hyperparameters)
├── run_main.py                     # Main experiment runner (single execution)
├── run_all_experiments.py          # Complete experimental suite
├── generate_figures.py             # Publication-quality figure generation
├── analyze_quantum_correlations.py # Quantum circuit analysis utilities
├── cleanup.sh                      # Script to clean up old results
├── git_push.sh                     # Git automation script
├── README.md                       # This file
│
├── data/                           # Input datasets
│   ├── cardiac_formatted_dataset-001.csv     # Primary cardiac dataset
│   ├── relevancy_ranked_genes_top_30.csv     # Top 30 genes by relevance
│   ├── relevancy_ranked_genes_top_50.csv     # Top 50 genes by relevance
│   └── relevancy_ranked_genes_top_100.csv    # Top 100 genes by relevance
│
├── logs/                           # Experiment logs and outputs
│
└── results/                        # Experimental results
    ├── data/                       # CSV and JSON results
    │   ├── all_results_*.json      # Aggregated results
    │   ├── exp_benchmark.csv       # Benchmark comparison results
    │   ├── exp_layer_depth.csv     # Layer depth experiment results
    │   ├── exp_qubit_scaling.csv   # Qubit scaling results
    │   └── *_params.npy            # Learned parameters
    ├── figures/                    # PNG publication-ready figures
    └── kernels/                    # Computed quantum kernel matrices (.npy)
```

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8+
- CUDA-Q (quantum computing framework)
- scikit-learn, numpy, pandas
- See [requirements.txt](../requirements.txt)

### 1. Clean Up Old Results (First Time Only)

```bash
cd CUDAQ
bash cleanup.sh
```

### 2. Run a Single Experiment

```bash
# Quick test (50 iterations, ~5 minutes)
python run_main.py --pca_dim 10 --n_layers 2 --max_iter 50

# Full experiment (300 iterations)
python run_main.py --pca_dim 20 --n_layers 2 --max_iter 300 --gene_set top50

# Custom configuration
python run_main.py --pca_dim 15 --n_layers 4 --max_iter 200 --shots 3000
```

### 3. Run Complete Experimental Suite

```bash
# Quick mode (basic configurations)
python run_all_experiments.py --quick

# Full mode (all configurations and repetitions)
python run_all_experiments.py --full
```

This runs three types of experiments:
- **Qubit Scaling**: 10, 20, 30 qubits
- **Layer Depth**: 1, 2, 4 layers
- **Benchmark**: QKA vs RBF, Linear, and RandomForest kernels

### 4. Generate Publication Figures

```bash
python generate_figures.py
```

Generates figures in `results/figures/` directory for analysis and publication.

---

## 📊 Experimental Results

### Publication Figures

| Figure | Description | Output File |
|--------|-------------|-------------|
| 1 | Quantum Kernel Matrix Visualization | `fig_01_kernel_matrix.png` |
| 2 | Qubit Scaling Performance | `fig_02_qubit_scaling.png` |
| 3 | Circuit Layer Depth Analysis | `fig_03_layer_depth.png` |
| 4 | Kernel Method Benchmark Comparison | `fig_04_benchmark.png` |
| 5 | Training Convergence Curves | `fig_05_convergence.png` |
| 6 | Prediction vs Ground Truth | `fig_06_prediction.png` |
| 7 | Circuit Weight Distribution | `fig_07_weights.png` |

### Key Metrics

- **R² Score**: Model coefficient of determination
- **MAE**: Mean Absolute Error for prediction accuracy
- **KTA**: Kernel Target Alignment (measures kernel-label correlation)
- **Training Time**: Computational cost in seconds
- **Parameters**: Total number of learnable circuit parameters

---

## 🔬 Experimental Design

### Experiment 1: Qubit Scaling

Evaluates how quantum circuit performance scales with increasing qubits (PCA dimensionality).

- **Configurations**: 10, 20, 30 qubits
- **Metrics**: R², MAE, Training Time
- **Purpose**: Identify optimal qubit count for cardiac gene expression

### Experiment 2: Circuit Layer Depth

Studies the impact of circuit depth on kernel expressiveness and model accuracy.

- **Configurations**: 1, 2, 4 entangling layers
- **Metrics**: R², KTA, Parameter Count
- **Purpose**: Balance model capacity and trainability

### Experiment 3: Kernel Method Benchmark

Compares QKA against classical kernel methods on cardiac gene prediction.

- **Methods**: QKA-SVR, RBF-SVR, Linear-SVR, Random Forest
- **Dataset**: Top 30, 50, and 100 genes
- **Metrics**: R², MAE, Training Time, Parameter Efficiency

---

## 🧬 Quantum Circuit Architecture

### Circuit Design

```
Input encoding layer (Hadamard + RZ gates):
q₀: ─H─RZ(θ₀)─────────────────....─M
q₁: ─H─RZ(θ₁)─────────────────....─M
q₂: ─H─RZ(θ₂)─────────────────....─M
q₃: ─H─RZ(θ₃)─────────────────....─M

Entangling layer (CNOT ladder):
      ─●─────────────....─
       │
      ─X──●──────────....─
          │
        ─X──●────────....─
            │
          ─X────────....─

Variational layer (RX rotations):
q₀: ─RX(θ₄)─RY(θ₈)─ ....
q₁: ─RX(θ₅)─RY(θ₉)─ ....
q₂: ─RX(θ₆)─RY(θ₁₀)─....
q₃: ─RX(θ₇)─RY(θ₁₁)─....
```

### Kernel Computation

The quantum kernel is computed as:

$$K(x, x') = |\langle 0|U^\dagger(x')U(x)|0\rangle|^2$$

where U(x) is the parameterized quantum circuit encoding input x, and the overlap is measured in computational basis.

### Parameter Count

Total learnable parameters: **n_qubits × n_layers × 2**

For a 10-qubit, 2-layer circuit: 40 parameters

---

## 📝 Command-Line Parameters

```bash
python run_main.py [OPTIONS]

Options:
  --pca_dim INT          PCA dimensionality (number of qubits)
                         Available: 10, 20, 30
                         [default: 10]
  
  --n_layers INT         Number of entangling layers
                         Available: 1, 2, 4
                         [default: 2]
  
  --max_iter INT         Maximum SVR optimization iterations
                         [default: 50]
  
  --shots INT            Quantum measurement shots per circuit evaluation
                         [default: 2000]
  
  --gene_set STR         Gene selection for prediction
                         Available: top30, top50, top100
                         [default: top30]

python run_all_experiments.py [OPTIONS]

Options:
  --quick                Run abbreviated experiment suite (faster)
  --full                 Run complete experiment suite (recommended)
```

---

## 💾 Output Files

### CSV Results Files
- `exp_qubit_scaling.csv`: Results across different qubit counts
- `exp_layer_depth.csv`: Results for various circuit depths
- `exp_benchmark.csv`: Comparison with classical baselines
- `all_results_*.json`: Complete result aggregation with timestamps

### Kernel Matrices
- `layer_1_K.npy`, `layer_2_K.npy`, `layer_4_K.npy`: Computed kernels for each layer depth
- `qubit_10_K.npy`, `qubit_20_K.npy`, `qubit_30_K.npy`: Kernels for each qubit configuration

### Learned Parameters
- `*_params.npy`: Optimized quantum circuit parameters

---

## 🔧 Configuration File

Edit [config.py](config.py) to customize:

```python
# Data paths
DATA_DIR = 'data/'
RESULT_DIR = 'results/'

# Quantum circuit parameters
DEFAULT_SHOTS = 2000
MAX_ITERATIONS = 300

# SVR hyperparameters
SVR_C = 1.0
SVR_EPSILON = 0.1

# Gene selection
TOP_GENES = 30  # or 50, 100
```

---

## 📚 Methodology

### Quantum Kernel Alignment

QKA is a hybrid quantum-classical approach that:

1. **Encodes data** into quantum pure states using parameterized circuits
2. **Computes quantum kernels** via quantum circuit overlap measurements
3. **Trains kernel parameters** to maximize kernel-label alignment
4. **Applies classical SVR** with the learned quantum kernel

### Advantages Over Classical Kernels

- **Adaptive expressiveness**: Circuit parameters learn problem-specific kernel structures
- **Scalability**: Quantum computation provides exponential expressiveness in theory
- **Hybrid efficiency**: Combines quantum and classical computing strengths

---

## 📈 Expected Results

On cardiac gene expression prediction:

- **QKA-SVR R² Score**: 0.75-0.85 (top 30 genes)
- **Compared to RBF**: ~5-15% improvement in prediction accuracy
- **Training time**: 2-10 minutes per experiment (GPU accelerated)
- **Kernel dimensionality**: Depends on qubit/layer configuration

Results vary based on random initialization and dataset sampling.

---

## 🐛 Troubleshooting

### CUDA-Q Not Found
```bash
# Ensure CUDA-Q environment is active
conda activate cudaq
python run_main.py --pca_dim 10
```

### Out of Memory
```bash
# Reduce shots or qubit dimension
python run_main.py --pca_dim 10 --shots 1000
```

### Import Errors
```bash
# Verify dependencies
pip install -r ../requirements.txt
```

---

## 📖 References

- **CUDA-Q Framework**: NVIDIA's quantum computing platform
- **Quantum Kernel Methods**: Huang et al., "Power of data in quantum machine learning" (2021)
- **Cardiac Gene Expression**: Application to beat rate prediction in cardiac tissue

---

## ✨ Authors & Contact

**Project Lead**: Leo  
**Email**: leo07010@gmail.com  
**Institution**: QKA-Cardiac Research Project

---

## 📄 License

This project is provided as-is for research purposes. 

---

## 🔄 Version History

- **v2.0** (Feb 2026): Complete experimental suite with benchmarks
- **v1.0** (Jan 2026): Initial QKA implementation
