#!/bin/bash
#SBATCH --job-name=ar
#SBATCH --output=logs/timechat/96f_long-context/%x_%j.out
#SBATCH --error=logs/timechat/96f_long-context/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --partition=fair_gpu                        

source ~/.bashrc
conda init
conda activate timechat

num_frames=96

echo "Running with $num_frames frames and long-context..."
python -m eval_ar --num-frames $num_frames --long-context --video-path /beegfs-scratch/speirone/ego4d_videos/ar --ann-path ego4d/annotations/v1/
