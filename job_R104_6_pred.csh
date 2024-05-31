#!/bin/bash

#SBATCH --account=def-alamj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=R104_6_pred
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm_R104_6_pred.out

source ~/jupEnv/bin/activate
python pred_R104_6.py