#!/bin/sh
#SBATCH -A m4287
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

module load conda
conda activate torchvenv
cd /global/homes/i/ibrahime/temp/OmniLearnLightining/scripts/


### Top Task

## Micro Model
# From Scratch
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01 --fine_tune=n --dataset=top --mode=classifier --use_one_cycle=n

# PreTrained
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01  --fine_tune=y --dataset=top --mode=classifier --use_one_cycle=y --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/micro/super-gen.ckpt
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01  --fine_tune=y --dataset=top --mode=classifier --use_one_cycle=y --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/micro/gen-only.ckpt
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01  --fine_tune=y --dataset=top --mode=classifier --use_one_cycle=y --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/micro/super-only.ckpt


## HPARAMS NOT SET
## Small Model
# From Scratch
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01 --fine_tune=n --dataset=top --mode=classifier --use_one_cycle=n

# PreTrained
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01  --fine_tune=y --dataset=top --mode=classifier --use_one_cycle=y --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/super-gen.ckpt
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01  --fine_tune=y --dataset=top --mode=classifier --use_one_cycle=y --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/gen-only.ckpt
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01  --fine_tune=y --dataset=top --mode=classifier --use_one_cycle=y --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/super-only.ckpt


## HPARAMS NOT SET
## Medium Model
# From Scratch
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=medium --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01 --fine_tune=n --dataset=top --mode=classifier --use_one_cycle=n

# PreTrained
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=medium --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01  --fine_tune=y --dataset=top --mode=classifier --use_one_cycle=y --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/medium/super-gen.ckpt
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=medium --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01  --fine_tune=y --dataset=top --mode=classifier --use_one_cycle=y --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/medium/gen-only.ckpt
srun --gpus-per-node 4 python3 train_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=medium --use_wandb --batch_size=256 --lr 1e-3 --weight_decay 0.01  --fine_tune=y --dataset=top --mode=classifier --use_one_cycle=y --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/medium/super-only.ckpt
