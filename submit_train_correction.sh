#!/bin/bash
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --job-name=chem-diff
#SBATCH --output=output_%j.txt
#SBATCH --gres=gpu:V100:1

bash scripts/run_train_correction.sh gdb13 1 False False False 5000