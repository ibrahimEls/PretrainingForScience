#!/bin/sh
#SBATCH -A m4287
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

module load conda
conda activate torchvenv
cd /global/homes/i/ibrahime/FoundationModelStudy/OmnilearnLightning

## Small Model
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=100_000_000 --num_workers=32 --num_nodes=8 --model_size=small --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=128

## Medium Model
#srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=100_000_000 --num_workers=32 --num_nodes=8 --model_size=medium --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=32
