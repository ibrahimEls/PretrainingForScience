#!/bin/sh
#SBATCH -A m4287
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH -N 10
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

module load conda
conda activate torchvenv
cd /global/homes/i/ibrahime/FoundationModelStudy/OmnilearnLightning
srun --gpus-per-node 4 python3 train_lighting.py --dataset_size=100_000_000 --num_workers=32 --num_nodes=1 --model_size=small --use_pid y --use_add y --epoch=1
