#!/bin/bash
#SBATCH --job-name=timechat_pnr                     # Job name
#SBATCH --output=logs/timechat/%x_%j.out            # Output log (%x=job-name, %j=job-id)
#SBATCH --error=logs/timechat/%x_%j.err             # Error log
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --cpus-per-task=24                          # Number of CPU cores per task
#SBATCH --mem=32G                                   # Total memory
#SBATCH --gres=gpu:1                                # Number of GPUs (remove if not needed)
#SBATCH --time=24:00:00                             # Time limit (hh:mm:ss)
#SBATCH --partition=fair_gpu                        

conda activate timechat

# Run the script

echo "Running with 8 frames and 0 icl examples..."
python -m eval_pnr --icl-examples 0 --num-frames 8

echo "Running with 16 frames and 0 icl examples..."
python -m eval_pnr --icl-examples 0 --num-frames 16

echo "Running with 32 frames and 0 icl examples..."
python -m eval_pnr --icl-examples 0 --num-frames 32
