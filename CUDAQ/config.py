"""
QKA 研究配置文件
================
集中管理所有實驗參數和路徑
"""
import os

# ==========================================
# 路徑設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Locate data directory: prefer `CUDAQ/data`, but fall back to a top-level `data`
def _find_data_dir(base_dir, max_up=4):
    cur = base_dir
    for _ in range(max_up):
        cand = os.path.join(cur, 'data')
        if os.path.isdir(cand):
            return cand
        cur = os.path.dirname(cur)
    # final fallback (may not exist) — keep original behaviour
    return os.path.join(base_dir, 'data')

DATA_DIR = _find_data_dir(BASE_DIR)
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 數據文件 (本地路徑)
MAIN_DATA_FILE = os.path.join(DATA_DIR, 'cardiac_formatted_dataset-001.csv')
GENE_FILES = {
    'top30': os.path.join(DATA_DIR, 'relevancy_ranked_genes_top_30.csv'),
    'top50': os.path.join(DATA_DIR, 'relevancy_ranked_genes_top_50.csv'),
    'top100': os.path.join(DATA_DIR, 'relevancy_ranked_genes_top_100.csv'),
}

# 結果輸出目錄
RESULT_DATA_DIR = os.path.join(RESULT_DIR, 'data')
RESULT_FIG_DIR = os.path.join(RESULT_DIR, 'figures')
RESULT_KERNEL_DIR = os.path.join(RESULT_DIR, 'kernels')

# ==========================================
# 量子計算參數
# ==========================================
DEFAULT_SHOTS = 2000          # 採樣次數
DEFAULT_MAX_ITER = 300        # COBYLA 最大迭代
DEFAULT_TOL = 1e-4            # 收斂閾值

# 實驗配置
QUBIT_CONFIGS = [10, 20]       # 測試的量子位元數 (PCA 維度)
LAYER_CONFIGS = [1, 2, 4]          # 測試的電路層數
ENCODING_TYPES = ['angle']         # 編碼方式

# 預設層數（方便 CLI 與程式內使用）
DEFAULT_N_LAYERS = LAYER_CONFIGS[0]

# ==========================================
# SVR 參數
# ==========================================
SVR_C = 100.0
SVR_EPSILON = 0.1

# ==========================================
# 視覺化參數
# ==========================================
FIG_DPI = 300
FIG_FORMAT = 'png'

# ==========================================
# 輔助函數
# ==========================================
def ensure_dirs():
    """確保結果目錄存在"""
    os.makedirs(RESULT_DATA_DIR, exist_ok=True)
    os.makedirs(RESULT_FIG_DIR, exist_ok=True)
    os.makedirs(RESULT_KERNEL_DIR, exist_ok=True)

def get_result_path(filename, subdir='data'):
    """取得結果文件路徑"""
    dirs = {
        'data': RESULT_DATA_DIR,
        'figures': RESULT_FIG_DIR,
        'kernels': RESULT_KERNEL_DIR
    }
    return os.path.join(dirs.get(subdir, RESULT_DIR), filename)

# 初始化目錄
ensure_dirs()
