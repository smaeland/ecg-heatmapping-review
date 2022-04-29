#!/bin/bash
#SBATCH -J eval-metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH -p xeongold16q
####SBATCH --gres=gpu:1
####SBATCH -p a100q 

module load pytorch-py37-cuda11.2-gcc8/1.9.1

srun python evaluate_model.py 
