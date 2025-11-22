#!/bin/sh
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
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

# --------- Config variables ---------
PRETRAIN_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Paths/Pre-Trained-Model-Paths.json"
LAYOUT_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Paths/Top-Class-Finetuned-Model-Paths-Template.json"
STATE_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Paths/Top-Class-Finetuned-Model-Paths-States.json"

DOWNSTREAM_DATASETS="all"
MODEL_SIZES="micro"
#PRE_TRAINING_MODES="classifier,generator,mpm,classifier+generator,classifier+mpm,generator+mpm,from_scratch"
PRE_TRAINING_MODES="classifier,generator,classifier+generator,from_scratch"
MODE="classifier"

FINETUNING_DATASET="top"
N_RUNS=3
BASE_SEED=1603
NODES=4

CMD_TEMPLATE_FILE="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/finetune-template.sh"
# -----------------------------------

export cmd="python finetune_lightning.py \
  --pretrain-json $PRETRAIN_JSON \
  --layout-json $LAYOUT_JSON \
  --state-json $STATE_JSON \
  --downstream-datasets $DOWNSTREAM_DATASETS \
  --model-sizes $MODEL_SIZES \
  --pre_training_modes $PRE_TRAINING_MODES \
  --mode $MODE \
  --n-runs $N_RUNS \
  --base-seed $BASE_SEED \
  --cmd-template-file $CMD_TEMPLATE_FILE\
  --dataset $FINETUNING_DATASET\
  --num_nodes=$NODES\
  --repo_dir=$REPO_DIR\
  --output_dir=$OUTPUT_DIR\
  --dataset_dir=$DATASET_PATH"
 # --dry-run <---- Turn on to see what will be run without running it

# slurm job:
srun --gpus-per-node 4 shifter bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && $cmd"

# Interactive use:
# (specify num_nodes=1 to avoid trying to use multiple nodes in interactive mode)
#shifter --image=docker:jobirk/omnilearn-lightning:v1.0.0 bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && $cmd"
