#!/bin/bash

#SBATCH --account=def-alamj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=R103_1_train
#SBATCH --gpus-per-node=1
#SBATCH --output=~/scratch/WAE/slurmOUT/slurm_R103_1_train.out

source ~/jupEnv/bin/activate
python ~/scratch/WAE/trains/train_R103_1.py
