#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpulab02
#SBATCH -J t2vec
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpulab02

source activate nvib
python baseline/t2vec/train.py
