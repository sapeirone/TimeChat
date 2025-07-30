#!/bin/bash
#SBATCH --job-name=timechat_mq_ZS                      # Job name
#SBATCH --output=logs/timechat/3007/%x_%j.out            # Output log (%x=job-name, %j=job-id)
#SBATCH --error=logs/timechat/3007/%x_%j.err             # Error log
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --cpus-per-task=24                          # Number of CPU cores per task
#SBATCH --mem=32G                                   # Total memory
#SBATCH --gres=gpu:1                                # Number of GPUs (remove if not needed)
#SBATCH --time=24:00:00                             # Time limit (hh:mm:ss)
#SBATCH --partition=fair_gpu                        

conda activate timechat

echo "Running with 8 frames..."
python -m eval_mq --num-frames 8 --video-path /beegfs-scratch/speirone/ego4d_videos/mq --ann-path ego4d/annotations/v1/