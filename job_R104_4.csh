#!/bin/bash

#SBATCH --account=def-alamj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=R104_4_train
#SBATCH --gpus-per-node=1

source ~/jupEnv/bin/activate
python train_R104_4.py