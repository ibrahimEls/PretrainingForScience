#!/bin/sh
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
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
# REPO_DIR="/global/homes/j/jobirk/repositories/OmniLearnLightning"
OUTPUT_DIR="/pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/"
# OUTPUT_DIR=/pscratch/sd/j/jobirk/omnilearned_output/omnilearn_output
DATASET_PATH="/pscratch/sd/i/ibrahime/datasets/"
# DATASET_PATH=/pscratch/sd/j/jobirk/omnilearn_datasets/
# ------------------------------------

MODEL_PATHS_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Top-Class-Finetuned-Model-Paths.json"
METRICS_PATHS_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Results/Top-Class-Finetuned-Model-Metrics.json"

FINETUNING_DATASET="top"
DATASET_SIZE=-1

TRAINING_CFG_ALL_SIZES=" --dataset=$FINETUNING_DATASET --dataset_size=$DATASET_SIZE --outdir=$OUTPUT_DIR --path=$DATASET_PATH --model_paths_json=$MODEL_PATHS_JSON --metrics_json=$METRICS_PATHS_JSON"

### Micro Model
TRAINING_CFG_SIZE_SPECIFIC="--num_workers=2 --num_nodes=4 --model_size=micro --lr 1e-3 --weight_decay 0.01 --batch_size=256"
### Small model
# TRAINING_CFG_SIZE_SPECIFIC="--num_workers=32 --num_nodes=8 --model_size=small --lr 5e-4 --weight_decay 0.3 --batch_size=128"
### Medium model
# TRAINING_CFG_SIZE_SPECIFIC="--num_workers=32 --num_nodes=32 --model_size=medium --lr 5e-6 --weight_decay 0.1 --batch_size=32"

TRAINING_CFG="$TRAINING_CFG_ALL_SIZES $TRAINING_CFG_SIZE_SPECIFIC"

export cmd="python3 eval_lightning.py $TRAINING_CFG"

# slurm job:
#srun --gpus-per-node 4 shifter bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && $cmd"

# Interactive use:
# (specify num_nodes=1 to avoid trying to use multiple nodes in interactive mode)
shifter --image=docker:jobirk/omnilearn-lightning:v1.0.0 bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && $cmd --num_nodes=1"
