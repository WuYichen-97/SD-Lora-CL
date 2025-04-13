#!/bin/bash
#SBATCH --job-name=LoRA-seed1         # 作业名称
#SBATCH --output=output_%j.log        # 输出文件，%j 会被替换为作业ID
#SBATCH --error=error_%j.log          # 错误输出文件
#SBATCH --partition=gpu_reque          # 分区名称
#SBATCH --gres=gpu:1                  # 请求 3 块 GPU
#SBATCH --mem=25000                   # 请求 16 GB 内存
#SBATCH --cpus-per-task=4             # 每个任务使用 4 个 CPU 核
#SBATCH --time=0-20:00                # 最长运行时间（天-小时:分钟）

# 输出一些信息用于调试
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# 加载必要的模块（例如 Python、CUDA 等）
source activate pt2.3.0_cuda12.1 

# 启动你的程序
python your_script.py --batch-size 128 --epochs 50
