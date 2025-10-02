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
srun --gpus-per-node 4 python3 benchmark_scaling.py  --outdir benchmarks/ --cases 8n4g  --dataset jetclass --path /pscratch/sd/i/ibrahime/datasets/  --batch_size 64 --num_workers 32 --model_size medium --warmup_steps 20 --measure_steps 150 --use_pid --use_add
