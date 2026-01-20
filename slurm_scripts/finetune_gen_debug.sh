#!/bin/sh
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --image=docker:jobirk/omnilearn-lightning:v1.0.3

# print this file content (so we have the job configuration saved in the logs)
cat $0

# ------- User-specific paths --------
REPO_DIR="/global/homes/j/jobirk/repositories/OmniLearnLightning_dev"
OUTPUT_DIR=/pscratch/sd/j/jobirk/omnilearned_output/omnilearn_output
DATASET_PATH=/pscratch/sd/j/jobirk/omnilearn_datasets_dev/
# ------------------------------------

# --------- Config variables ---------
# Toggle which JetNet dataset to use: "jetnet150" or "jetnet30"
FINETUNING_DATASET="jetnet150"

# Set paths based on dataset selection
if [ "$FINETUNING_DATASET" = "jetnet150" ]; then
    LAYOUT_JSON="${REPO_DIR}/assets/Paths/JetNet150-Gen-Finetuned-Model-Paths-Template.json"
    STATE_JSON="${REPO_DIR}/assets/Paths/JetNet150-Gen-Finetuned-Model-Paths-States.json"
elif [ "$FINETUNING_DATASET" = "jetnet30" ]; then
    LAYOUT_JSON="${REPO_DIR}/assets/Paths/JetNet30-Gen-Finetuned-Model-Paths-Template.json"
    STATE_JSON="${REPO_DIR}/assets/Paths/JetNet30-Gen-Finetuned-Model-Paths-States.json"
else
    echo "Error: FINETUNING_DATASET must be 'jetnet150' or 'jetnet30'"
    exit 1
fi

PRETRAIN_JSON="${REPO_DIR}/assets/Paths/Pre-Trained-Model-Paths.json"

# DOWNSTREAM_DATASETS="all"  # can also be e.g. "10k-JetNet150" or "1k-JetNet150,10k-JetNet150,100k-JetNet150"
DOWNSTREAM_DATASETS="100k-JetNet150"  # can also be e.g. "10k-JetNet150" or "1k-JetNet150,10k-JetNet150,100k-JetNet150"
MODEL_SIZES="micro"
# PRE_TRAINING_MODES="from_scratch,classifier,classifier+generator"  # comma-separated list list "from_scratch,classifier,classifier+generator"
PRE_TRAINING_MODES="from_scratch"  # comma-separated list list "from_scratch,classifier,classifier+generator"
MODE="generator"

N_RUNS=1
BASE_SEED=1603
NODES=1

CMD_TEMPLATE_FILE="${REPO_DIR}/assets/finetune-gen-template.sh"
# -----------------------------------

export cmd="python3 finetune_lightning.py \
  --pretrain-json $PRETRAIN_JSON \
  --layout-json $LAYOUT_JSON \
  --state-json $STATE_JSON \
  --downstream-datasets $DOWNSTREAM_DATASETS \
  --model-sizes $MODEL_SIZES \
  --pre_training_modes $PRE_TRAINING_MODES \
  --mode $MODE \
  --n-runs $N_RUNS \
  --base-seed $BASE_SEED \
  --cmd-template-file $CMD_TEMPLATE_FILE \
  --dataset $FINETUNING_DATASET \
  --num_nodes=$NODES \
  --repo_dir=$REPO_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_dir=$DATASET_PATH \
  --dry-run" # <---- Turn on to see what will be run without running it

module load conda
conda activate torchvenv
cd $REPO_DIR/scripts/ && $cmd
