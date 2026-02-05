#!/bin/bash
# Git 推送腳本
# 請在你的電腦上執行此腳本

echo "=== QKA Git Push Script ==="

# 1. 清理舊檔案
echo "[1/5] Cleaning old files..."
rm -f QKA_exp.py QKA_lib.py qka_core.py quick_test_demo.py
rm -f ana_*.py exp_*.py
rm -f run_all.sh run_all_execution.log pid.txt
rm -f QKA_Test.ipynb
rm -rf result/ __pycache__/

# 2. 初始化 Git (如果需要)
if [ ! -d ".git" ]; then
    echo "[2/5] Initializing git..."
    git init
    git branch -m main
else
    echo "[2/5] Git already initialized"
    # 移除可能的 lock 文件
    rm -f .git/index.lock
fi

# 3. 設定 Git 用戶
git config user.email "leo07010@gmail.com"
git config user.name "Leo"

# 4. 添加檔案
echo "[3/5] Adding files..."
git add config.py
git add run_main.py
git add run_all_experiments.py
git add generate_figures.py
git add README.md
git add cleanup.sh
git add .gitignore
git add data/

# 5. 提交
echo "[4/5] Committing..."
git commit -m "Restructure QKA codebase for thesis

- Clean modular architecture (config.py, run_main.py)
- Support for 1/2/4 layer circuits
- Experiments: qubit scaling, layer depth, benchmark
- Auto figure generation
- Local data folder structure

Co-Authored-By: Claude <noreply@anthropic.com>"

# 6. 推送
echo "[5/5] Pushing to GitHub..."
echo ""
echo "請選擇推送方式:"
echo "  A) 推送到現有 repo: git remote add origin <your-repo-url> && git push -u origin main"
echo "  B) 創建新 repo: gh repo create QKA-Cardiac --public --source=. --push"
echo ""
echo "或手動執行:"
echo "  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
echo "  git push -u origin main"
