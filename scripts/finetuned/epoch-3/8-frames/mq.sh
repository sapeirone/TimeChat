#!/bin/bash
#SBATCH --job-name=timechat_mq
#SBATCH --output=logs/timechat/finetuned/8f_epoch3/%x_%j.out
#SBATCH --error=logs/timechat/finetuned/8f_epoch3/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --partition=fair_gpu                        

source ~/.bashrc
conda init
conda activate timechat

echo "Running with 8 frames..."
python -m eval_mq_ft --num-frames 8 --video-path /beegfs-scratch/speirone/ego4d_videos/mq --ann-path ego4d/annotations/v1/ --timechat-ckpt timechat/ckpt/timechat/train_stage2_llama2_7b_instruct12.ego4d_it.fix-mq/20250730201/checkpoint_11.pth