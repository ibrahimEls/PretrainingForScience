#!/bin/sh
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 36:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

# print this file content (so we have the job configuration saved in the logs)
cat $0

# ------- User-specific paths --------
REPO_DIR="/global/homes/i/ibrahime/temp/OmniLearnLightining/"
OUTPUT_DIR="/pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/"
DATASET_PATH="/pscratch/sd/i/ibrahime/datasets/"
# ------------------------------------

# --------- Required user inputs ---------
# Directory containing per-val-check pre-training checkpoints (filenames must
# contain 'val_loss=').
PRETRAIN_CKPT_DIR="/pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/_medium_classifier_jetclass_dataset_-1_ibrahime/v10_medium_classifier_jetclass_dataset_-1_ibrahime/checkpoints/"
#PRETRAIN_CKPT_DIR="/pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/_medium_classifier+generator_jetclass_dataset_-1_ibrahime/v20_medium_classifier+generator_jetclass_dataset_-1_ibrahime/checkpoints/"
#PRETRAIN_CKPT_DIR="/pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/_medium_classifier+generator_jetclass_dataset_-1_ibrahime/v22_medium_classifier+generator_jetclass_dataset_-1_ibrahime/checkpoints/"
# Where to copy best fine-tuned ckpts (one subdir per pretrain step).
DEST_DIR="/global/cfs/cdirs/m3246/Omnilearned_Study/PretrainStepSweep/Medium/classifier/"

# Output state JSON describing every fine-tune run.
STATE_JSON="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/Paths/Classifier-PretrainStepSweep-States.json"

# Take every Nth val checkpoint (1 = all).
INTERVAL=4

# Bookkeeping (size-dependent hparams come from MODEL_HPARAMS in the python script).
MODEL_SIZE="medium"
PRE_TRAINING_MODE="classifier+generator"
MODE="classifier"

# Downstream-side knobs.
FINETUNING_DATASET="top"
DOWNSTREAM_KEY="1M-Top"
DOWNSTREAM_SHOTS=-1

SEED=0
NODES=4

CMD_TEMPLATE_FILE="/global/homes/i/ibrahime/temp/OmniLearnLightining/assets/finetune-template.sh"
# ----------------------------------------

export cmd="python3 finetune_pretrain_sweep.py \
  --pretrain-ckpt-dir $PRETRAIN_CKPT_DIR \
  --dest-dir $DEST_DIR \
  --state-json $STATE_JSON \
  --cmd-template-file $CMD_TEMPLATE_FILE \
  --repo_dir=$REPO_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_dir=$DATASET_PATH \
  --dataset $FINETUNING_DATASET \
  --mode $MODE \
  --model_size $MODEL_SIZE \
  --pre_training_mode $PRE_TRAINING_MODE \
  --num_nodes=$NODES \
  --interval $INTERVAL \
  --seed $SEED \
  --downstream-key $DOWNSTREAM_KEY \
  --downstream-shots $DOWNSTREAM_SHOTS"
  # --dry-run" # <---- Turn on to see what will be run without running it

module load conda
conda activate torchvenv
cd $REPO_DIR/scripts/ && $cmd
