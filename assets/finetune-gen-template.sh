#!/bin/sh

REPO_DIR="{repo_dir}"
OUTPUT_DIR="{output_dir}"
DATASET_PATH="{dataset_dir}"

PRETRAINED_CKPT="{pretrain_ckpt}"
FINETUNING_DATASET="{dataset}"

TOKENIZER_CKPT="${{REPO_DIR}}/assets/kmeans_model_2025-12-17-19-08-10-Dry-Gondola_noAddInfo_8192_codes.pth"  # <-- kinematic features only
DATASET_SIZE="{downstream_shots}"

POS_ENCODING_TYPE="sort_descending_in_masked_subset"
MASKING_FRACTION=0.4

VAL_CHECK=1
PATIENCE=100
MAX_EPOCHS=500

# For finetuning vs from-scratch
# JetNet-specific (important that this is correct in args below): --num_classes=5 --use_pid=n --use_add=n
TRAINING_CFG_ALL_SIZES="--use_wandb y --wandb_project omnilearned --wandb_entity joschka-birk --limit_val_batches=1.0 --val_check_interval=${{VAL_CHECK}} --dataset=${{FINETUNING_DATASET}} --dataset_size=${{DATASET_SIZE}} {fine_tune_flag} {ckpt_flag} --use_pid n --use_add n --num_classes=5 --epoch=$MAX_EPOCHS --outdir=${{OUTPUT_DIR}} --path=${{DATASET_PATH}} --shuffle_val_test_indices y --seed_for_initial_shuffling={seed} --patience=${{PATIENCE}} --pos_encoding_type=${{POS_ENCODING_TYPE}} --masking_fraction=${{MASKING_FRACTION}} --use_one_cycle y --save_top_k -1"

# Size-specific hyperparams (Python fills these)
TRAINING_CFG_SIZE_SPECIFIC="--num_workers={num_workers} --num_nodes={num_nodes} --model_size={model_size} --lr {lr} --weight_decay {weight_decay} --batch_size={batch_size} --wandb_tag={dataset}-gen-finetuning"

TRAINING_CFG="${{TRAINING_CFG_ALL_SIZES}} ${{TRAINING_CFG_SIZE_SPECIFIC}}"

# Final command
srun --gpus-per-node 4 shifter --image=docker:jobirk/omnilearn-lightning:v1.0.3 bash -c "cd $REPO_DIR/scripts/ && source /opt/conda/bin/activate && export PYTHONPATH=${{REPO_DIR}}:$PYTHONPATH && python3 train_lightning.py $TRAINING_CFG --mode={mode}"
