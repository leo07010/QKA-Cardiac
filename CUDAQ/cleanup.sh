#!/bin/bash
# 清理舊檔案腳本
# 執行方式: bash cleanup.sh

echo "=== 清理 CUDAQ 資料夾中的舊檔案 ==="

# 刪除舊的 Python 檔案
rm -f QKA_exp.py QKA_lib.py qka_core.py quick_test_demo.py
rm -f ana_01_performance_boxplot.py ana_02_kernel_heatmap.py
rm -f ana_03_convergence_plot.py ana_04_feature_importance.py ana_05_stats_test.py
rm -f exp_01_benchmark_comparison.py exp_02_ablation_depth.py
rm -f exp_03_ablation_encoding.py exp_04_pca_scaling.py

# 刪除舊的其他檔案
rm -f run_all.sh run_all_execution.log pid.txt
rm -f QKA_Test.ipynb

# 刪除舊的資料夾
rm -rf result/ __pycache__/

echo "=== 清理完成 ==="
echo ""
echo "保留的檔案:"
echo "  - config.py          (配置文件)"
echo "  - run_main.py        (主程式)"
echo "  - run_all_experiments.py (完整實驗)"
echo "  - generate_figures.py    (圖表生成)"
echo "  - README.md          (說明文件)"
echo "  - data/              (數據資料夾)"
echo "  - results/           (結果資料夾)"
