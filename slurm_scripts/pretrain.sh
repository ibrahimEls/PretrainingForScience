#!/bin/sh
#SBATCH -A m4287
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export REPO_DIR="/global/homes/i/ibrahime/temp/OmniLearnLightining/"

### Micro Model
#export cmd="python3 train_lightning.py --mode=classifier --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --lr 1e-3 --weight_decay 0.01 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=256"
#export cmd="python3 train_lightning.py --mode=generator --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --lr 1e-3 --weight_decay 0.01 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=256"
#export cmd="python3 train_lightning.py --mode=mpm --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --lr 1e-3 --weight_decay 0.01 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=256"
export cmd="python train_lightning.py --mode=classifier+generator --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --lr 1e-3 --weight_decay 0.01 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=256"
#export cmd="python3 train_lightning.py --mode=classifier+mpm --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --lr 1e-3 --weight_decay 0.01 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=256"
#export cmd="python3 train_lightning.py --mode=generator+mpm --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --lr 1e-3 --weight_decay 0.01 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=256"
#export cmd="python3 train_lightning.py --mode=pretrain --dataset_size=-1 --num_workers=2 --num_nodes=4 --model_size=micro --lr 1e-3 --weight_decay 0.01 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=256"

### Small Model
#export cmd="python3 train_lightning.py --mode=classifier --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --lr 5e-4 --weight_decay 0.3 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=128"
#export cmd="python3 train_lightning.py --mode=generator --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --lr 5e-4  --weight_decay 0.3 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=128"
#export cmd="python3 train_lightning.py --mode=mpm --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --lr 5e-4  --weight_decay 0.3 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=128"
#export cmd="python3 train_lightning.py --mode=classifier+generator --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --lr 5e-4  --weight_decay 0.3 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=128"
#export cmd="python3 train_lightning.py --mode=classifier+mpm --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --lr 5e-4  --weight_decay 0.3 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=128"
#export cmd="python3 train_lightning.py --mode=generator+mpm --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --lr 5e-4  --weight_decay 0.3 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=128"
#export cmd="python3 train_lightning.py --mode=pretrain --dataset_size=-1 --num_workers=32 --num_nodes=8 --model_size=small --lr 5e-4 --weight_decay 0.3 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=128"


### Medium Model
#export cmd="python3 train_lightning.py --mode=classifier --dataset_size=-1 --num_workers=32 --num_nodes=32--model_size=medium --lr 5e-6 --weight_decay 0.1 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=32"
#export cmd="python3 train_lightning.py --mode=generator --dataset_size=-1 --num_workers=32 --num_nodes=32--model_size=medium --lr 5e-6 --weight_decay 0.1 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=32"
#export cmd="python3 train_lightning.py --mode=mpm --dataset_size=-1 --num_workers=32 --num_nodes=32--model_size=medium --lr 5e-6 --weight_decay 0.1 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=32"
#export cmd="python3 train_lightning.py --mode=classifier+generator --dataset_size=-1 --num_workers=32 --num_nodes=32--model_size=medium --lr 5e-6 --weight_decay 0.1 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=32"
#export cmd="python3 train_lightning.py --mode=classifier+mpm --dataset_size=-1 --num_workers=32 --num_nodes=32--model_size=medium --lr 5e-6 --weight_decay 0.1 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=32"
#export cmd="python3 train_lightning.py --mode=generator+mpm --dataset_size=-1 --num_workers=32 --num_nodes=32--model_size=medium --lr 5e-6 --weight_decay 0.1 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=32"
#export cmd="python3 train_lightning.py --mode=pretrain --dataset_size=-1 --num_workers=32 --num_nodes=32--model_size=medium --lr 5e-6 --weight_decay 0.1 --use_pid y --use_add y --epoch=500 --use_wandb --batch_size=32"


srun --gpus-per-node 4  shifter bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && $cmd"

#shifter --image=docker:jobirk/omnilearned:latest bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && $cmd"
