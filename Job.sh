#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpulab02
#SBATCH -J LCSS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpulab02

source activate nvib
python experiment/cityemd/traj_simi.py -m LCSS -d Porto -hi 11 -t ../data/Porto/timeData/12_18.csv -hd ../data/Porto/timeData -b 16
