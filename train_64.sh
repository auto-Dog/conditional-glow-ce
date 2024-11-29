#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1

python train.py --prefix s64_K3_b8 --x_size (3,64,64) --y_size (3,64,64) --x_hidden_size 64