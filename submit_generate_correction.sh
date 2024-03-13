#!/bin/bash
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --job-name=chem-diff
#SBATCH --output=output_%j.txt
#SBATCH --gres=gpu:V100:1

CUDA_VISIBLE_DEVICES=1 && bash scripts/run_generate_correction.sh ckpts/gdb13/ema_0.9999_200000.pt 2000 ckpts/gdb13/ema_0.9999_200000.pt.samples_1000.steps-2000.clamp-no_clamp.txt