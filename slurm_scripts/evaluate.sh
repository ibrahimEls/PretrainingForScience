#!/bin/sh
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
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
STATES_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Paths/Top-Class-Finetuned-Model-Paths-States.json"
EVAL_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Results/Top-Class-Finetuned-Model-Paths-Eval.json"

DATASET="top"
GPU_ID=0
TASK="top_tagging"


shifter --image=docker:jobirk/omnilearn-lightning:v1.0.0 bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && \
  python eval_lightning.py \
  --states_json $STATES_JSON \
  --eval_json $EVAL_JSON \
  --dataset $DATASET \
  --path $DATASET_PATH \
  --outdir $OUTPUT_DIR \
  --gpuID $GPU_ID \
  --task $TASK"
