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
cd /global/homes/i/ibrahime/temp/OmniLearnLightining/scripts/


### Top Task

## Micro Model
# From Scratch
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=8 --model_size=micro --use_wandb --batch_size=256 --from_scratch
# PreTrained
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=8 --model_size=micro --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/micro/super-gen.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=8 --model_size=micro --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/smmicroall/gen-only.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=8 --model_size=micro --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/micro/super-only.ckpt

## Small Model
# From Scratch
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --use_wandb --batch_size=256 --from_scratch
# PreTrained
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/super-gen.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/gen-only.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/super-only.ckpt

srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/super-gen.ckpt


## Medium Model
# From Scratch
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=medium --use_wandb --batch_size=48 --from_scratch
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=medium --use_wandb --batch_size=48 --resume --ckpt /pscratch/sd/i/ibrahime/checkpoints/super_gen_medium_top_tagging_-1_top_-1-shot/version_13/checkpoints/last.ckpt

# PreTrained
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=medium --use_wandb --batch_size=32 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/medium/super-gen.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=medium --use_wandb --batch_size=32 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/medium/gen-only.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=medium --use_wandb --batch_size=32 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/medium/super-only.ckpt
