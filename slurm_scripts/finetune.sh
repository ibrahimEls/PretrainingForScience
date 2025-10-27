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
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --from_scratch
# PreTrained
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/micro/super-gen.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/micro/gen-only.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/micro/super-only.ckpt

#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_gen_only/gen_only_micro_1000000/version_4/checkpoints/last.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_only/super_only_micro_1000000/version_8/checkpoints/last.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/super_gen_micro_1000000/version_10/checkpoints/last.ckpt


## Small Model
# From Scratch
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --from_scratch
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --from_scratch --lr 5e-5 --weight_decay 0.5

# PreTrained
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --lr 1e-6 --weight_decay 0.5 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/super-only.ckpt
srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --lr 1e-6 --weight_decay 0.5 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/super-gen.ckpt


#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/super-gen.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/gen-only.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=4 --model_size=small --use_wandb --batch_size=256 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/small/super-only.ckpt


## Medium Model
# From Scratch
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=20 --model_size=medium --use_wandb --batch_size=48 --from_scratch --epoch 300
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=20 --model_size=medium --use_wandb --batch_size=48 --resume --ckpt /pscratch/sd/i/ibrahime/checkpoints/super_gen_medium_top_tagging_-1_top_-1-shot/version_13/checkpoints/last.ckpt

# PreTrained
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=20 --model_size=medium --use_wandb --batch_size=48 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/medium/super-gen.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=20--model_size=medium --use_wandb --batch_size=48 -ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/medium/gen-only.ckpt
#srun --gpus-per-node 4 python3 finetune_lightning.py --dataset_size=-1 --num_workers=32 --num_nodes=20 --model_size=medium --use_wandb --batch_size=48 --ckpt /pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/medium/super-only.ckpt
