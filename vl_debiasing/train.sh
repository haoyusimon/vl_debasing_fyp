#!/bin/sh
#SBATCH --job-name=FYP
#SBATCH --gpus=1
#SBATCH --partition=long
#SBATCH --nodelist=xgph6
#SBATCH --cpus-per-task=2
#SBATCH --time=5-00:00:00
#SBATCH --output=./logs/stdout_%j.log
#SBATCH --error=./logs/stderr_%j.log

CUDA_VISIBLE_DEVICES=0 python -u train_age.py --version age