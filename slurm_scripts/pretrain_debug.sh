#!/bin/sh
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --image=docker:jobirk/omnilearn-lightning:v1.0.0

# print this file content (so we have the job configuration saved in the logs)
cat $0

# ------- User-specific paths --------
REPO_DIR="/global/homes/i/ibrahime/temp/OmniLearnLightining/"
#REPO_DIR="/global/homes/j/jobirk/repositories/OmniLearnLightning_dev"
OUTPUT_DIR="/pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/"
#OUTPUT_DIR=/pscratch/sd/j/jobirk/omnilearned_output/omnilearn_output
DATASET_PATH="/pscratch/sd/i/ibrahime/datasets/"
#DATASET_PATH=/pscratch/sd/j/jobirk/omnilearn_datasets/
# ------------------------------------

# TOKENIZER_CKPT="${REPO_DIR}/assets/kmeans_model_2025-11-01-00-13-29-Idiotic-Name_withAddInfo_16384_codes.pth"  # <-- with trajectory displacement features
TOKENIZER_CKPT="${REPO_DIR}/assets/kmeans_model_2025-12-07-20-13-56-Funky-Desktop_noAddInfo_8192_codes.pth"  # <-- kinematic features only
DATASET_SIZE=-1
# POS_ENCODING_TYPE="sort_descending_all"
POS_ENCODING_TYPE="sort_descending_in_masked_subset"
# POS_ENCODING_TYPE="None"
MASKING_FRACTION=0.4

VAL_CHECK=5000 # 5000  # Set to one if DATASET_SIZE!=-1 else 5000
PATIENCE=10 # 4  # Set to 10 if DATASET_SIZE!=-1 else 4

PRETRAINED_CKPT="/pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/_micro_classifier_jetclass_dataset_-1_ibrahime/v3_micro_classifier_jetclass_dataset_-1_ibrahime/checkpoints/last.ckpt"
RESUME=y


TRAINING_CFG_ALL_SIZES="--use_wandb y --wandb_project omnilearned --limit_val_batches=250 --val_check_interval=$VAL_CHECK --dataset_size=$DATASET_SIZE --use_pid y --use_add y --epoch=500  --outdir=$OUTPUT_DIR --path=$DATASET_PATH --shuffle_val_test_indices y --seed_for_initial_shuffling=1603 --patience=$PATIENCE  --pos_encoding_type=$POS_ENCODING_TYPE --masking_fraction=$MASKING_FRACTION --resume=$RESUME --ckpt=$PRETRAINED_CKPT"

### Micro Model
TRAINING_CFG_SIZE_SPECIFIC="--num_workers=2 --num_nodes=4 --model_size=micro --lr 1e-3 --weight_decay 0.01 --batch_size=256"
### Small model
# TRAINING_CFG_SIZE_SPECIFIC="--num_workers=32 --num_nodes=4 --model_size=small --lr 5e-4 --weight_decay 0.3 --batch_size=128"
### Medium model
# TRAINING_CFG_SIZE_SPECIFIC="--num_workers=32 --num_nodes=4 --model_size=medium --lr 5e-6 --weight_decay 0.1 --batch_size=32"

TRAINING_CFG="$TRAINING_CFG_ALL_SIZES $TRAINING_CFG_SIZE_SPECIFIC"

export cmd="python3 train_lightning.py $TRAINING_CFG --mode=classifier"
# export cmd="python3 train_lightning.py $TRAINING_CFG --mode=generator"
# export cmd="python3 train_lightning.py $TRAINING_CFG --mode=mpm --tokenizer_ckpt=$TOKENIZER_CKPT"
# export cmd="python3 train_lightning.py $TRAINING_CFG --mode=classifier+generator"
# export cmd="python train_lightning.py $TRAINING_CFG --mode=classifier+mpm --tokenizer_ckpt=$TOKENIZER_CKPT"
# export cmd="python3 train_lightning.py $TRAINING_CFG --mode=generator+mpm --tokenizer_ckpt=$TOKENIZER_CKPT"
# export cmd="python3 train_lightning.py $TRAINING_CFG --mode=pretrain --tokenizer_ckpt=$TOKENIZER_CKPT"
# MPM with regression target
# export cmd="python train_lightning.py $TRAINING_CFG --mode=mpm"
# export cmd="python train_lightning.py $TRAINING_CFG --mode=classifier+mpm"

# slurm job:
srun --gpus-per-node 4 shifter bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && $cmd"

# Interactive use:
# (specify num_nodes=1 to avoid trying to use multiple nodes in interactive mode)
# shifter --image=docker:jobirk/omnilearn-lightning:v1.0.0 bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && $cmd --num_nodes=1"
