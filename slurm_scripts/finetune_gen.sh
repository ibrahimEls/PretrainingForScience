#!/bin/bash

# print this file content (so we have the job configuration saved in the logs)
cat $0

# ------- User-specific paths --------
REPO_DIR="/global/homes/j/jobirk/repositories/OmniLearnLightning"
OUTPUT_DIR=/pscratch/sd/j/jobirk/omnilearned_output/omnilearn_output
DATASET_PATH=/pscratch/sd/j/jobirk/omnilearn_datasets/

PRETRAINED_CKPT="{{pretrained_ckpt}}"
DATASET_SIZE="{{dataset_size}}"
RUN_DIR="{{run_dir}}"
FINE_TUNE="{{fine_tune}}"
SEED="{{seed}}"

# Hyperparameters (to be adjusted at job creation time):
MODEL_SIZE="{{model_size}}"
LR="{{lr}}"
LR_FACTOR="{{lr_factor}}"
WEIGHT_DECAY="{{weight_decay}}"
BATCH_SIZE="{{batch_size}}"
NUM_WORKERS="{{num_workers}}"
OPTIMIZER_TYPE="{{optimizer_type}}"

MAX_TRAINING_STEPS="{{max_training_steps}}"
VAL_CHECK_INTERVAL="{{val_check_interval}}"
WANDB_ENTITY="{{wandb_entity}}"
NODES="{{nodes}}"
IS_LOCAL_SINGLE_NODE_RUN="{{is_local_single_node_run}}"

TRAINING_CFG_ALL_SIZES="\
    --run_dir=$RUN_DIR \
    --model_size=$MODEL_SIZE \
    --mode=generator \
    --use_wandb y \
    --wandb_project omnilearned \
    --limit_val_batches=250 \
    --wandb_entity $WANDB_ENTITY \
    --seed=$SEED \
    --dataset=jetnet150 \
    --num_classes=5 \
    --fine_tune=$FINE_TUNE \
    --use_add=n \
    --use_pid=n \
    --max_training_steps=$MAX_TRAINING_STEPS \
    --optimizer_type=$OPTIMIZER_TYPE \
    --scheduler_total_steps=$MAX_TRAINING_STEPS \
    --patience=-1 \
    --save_top_k=-1 \
    --epoch=100000 \
    --use_cond=y \
    --check_val_every_n_epoch=$VAL_CHECK_INTERVAL \
    --dataset_size=$DATASET_SIZE \
    --outdir=$OUTPUT_DIR \
    --path=$DATASET_PATH \
    --shuffle_val_test_indices=y \
    --seed_for_initial_shuffling=777 \
    --treat_gen_pidembed_and_condembed_as_last_layer=y \
    --ckpt=$PRETRAINED_CKPT \
    --num_workers=$NUM_WORKERS \
    --lr=$LR \
    --lr_factor=$LR_FACTOR \
    --weight_decay=$WEIGHT_DECAY \
    --batch_size=$BATCH_SIZE"


TRAINING_CFG="$TRAINING_CFG_ALL_SIZES $TRAINING_CFG_SIZE_SPECIFIC"

export cmd="cd $REPO_DIR/scripts/ \
    && source /opt/conda/bin/activate \
    && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH \
    && python3 train_lightning.py $TRAINING_CFG"

if [ "$IS_LOCAL_SINGLE_NODE_RUN" = "y" ]; then
    # Run interactively with srun across 4 nodes (inside an interactive allocation)
    echo "Running interactively with srun across 4 nodes"
    srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 shifter --image=docker:jobirk/omnilearn-lightning:v1.0.4 bash -c "$cmd --num_nodes=4"
else
    # Run with slurm (production mode)
    echo "Running with srun across $NODES nodes"
    srun --gpus-per-node 4 shifter --image=docker:jobirk/omnilearn-lightning:v1.0.4 bash -c "$cmd --num_nodes=$NODES"
fi
