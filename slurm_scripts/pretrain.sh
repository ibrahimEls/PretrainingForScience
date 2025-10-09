#!/bin/sh
#SBATCH -A m4287
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

module load conda
conda activate torchvenv
cd /global/homes/i/ibrahime/temp/OmniLearnLightining/scripts/

### Micro Model
# Super-Gen
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=256 --pretraining_mode=super-gen

# Super-Only
#srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=256 --pretraining_mode=super-only

# Gen-Only
#srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=256 --pretraining_mode=gen-only


### Small Model
# Super-Gen
#srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=32 --model_size=small --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=128 --pretraining_mode=super-gen

# Super-Only
#srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=32 --model_size=small --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=128 --pretraining_mode=super-only

# Gen-Only
#srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=32 --model_size=small --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=128 --pretraining_mode=gen-only

### Medium Model

# Super-Gen
#srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=32 --model_size=medium --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=32 --pretraining_mode=super-gen

# Super-Only
#srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=32 --model_size=medium --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=32 --pretraining_mode=super-only

# Gen-Only
#srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=32 --model_size=medium --use_pid y --use_add y --epoch=1 --use_wandb --batch_size=32 --pretraining_mode=gen-only
