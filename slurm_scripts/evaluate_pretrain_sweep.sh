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
#SBATCH --image=docker:jobirk/omnilearn-lightning:v1.0.4

# print this file content (so we have the job configuration saved in the logs)
cat $0

# ------- User-specific paths --------
REPO_DIR="/global/homes/i/ibrahime/temp/OmniLearnLightining/"
OUTPUT_DIR="/pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/"
DATASET_PATH="/pscratch/sd/i/ibrahime/datasets/"
# ------------------------------------

# --------- Required user inputs ---------
STATE_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Paths/Classifier-PretrainStepSweep-States.json"
#STATE_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Paths/Classifier+Generator-PretrainStepSweep-States.json"
EVAL_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Results/Classifier-PretrainStepSweep-Eval.json"
#EVAL_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Results/Classifier+Generator-PretrainStepSweep-Eval.json"

DATASET="top"
GPU_ID=0
TASK="top_tagging"
# ----------------------------------------

shifter --image=docker:jobirk/omnilearn-lightning:v1.0.4 bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && \
  python eval_pretrain_sweep.py \
  --state-json $STATE_JSON \
  --eval-json $EVAL_JSON \
  --dataset $DATASET \
  --path $DATASET_PATH \
  --outdir $OUTPUT_DIR \
  --gpuID $GPU_ID \
  --task $TASK"
  # --dry-run" # <---- Turn on to see what will be run without running it
